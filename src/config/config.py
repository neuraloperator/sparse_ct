import importlib
import pathlib
import torch
from fvcore.common.config import CfgNode as CN
from src.engines import TrainVal, VarnetLightningModule
import re
from torch.serialization import safe_globals
from fvcore.common.config import CfgNode
from src.engines import Test


class Configurator:
    def __init__(self, args):
        """
        Create configs and make fixes
        """
        self.cfg = CN(CN.load_yaml_with_base(args.config))

        self._check_cfg()

        if args.fix is not None:
            for k in args.fix[::2]:
                parts = k.split(".")
                node = self.cfg
                for part in parts[:-1]:
                    if not hasattr(node, part):
                        node[part] = CN()
                    node = node[part]

                if parts[-1] not in node:
                    node[parts[-1]] = None

            self.cfg.merge_from_list(args.fix)

        self.cfg = self._default_fix(self.cfg)

        self._check_cfg()
        self.cfg.freeze()

    def _check_cfg(self):
        if 'init_modules' in self.cfg.keys():
            allowed = {'sampler', 'sino_reconstructor', 'image_reconstructor',
                       'predictor', 'reconstructor'}  # 'reconstructor' kept for backward compat
            assert set(
                self.cfg.init_modules) <= allowed, f'init_modules must be a subset of {allowed}!'

    def _default_fix(self, cfg):
        if 'exp_dir' not in cfg.keys():
            cfg.exp_dir = f"results/{cfg.exp_name}"
        cfg.logger.name = cfg.exp_name
        cfg.logger.save_dir = cfg.exp_dir
        cfg.trainer.default_root_dir = cfg.exp_dir
        return cfg

    @staticmethod
    def str_to_class(module_name, class_name):
        """Return a class instance from a string reference"""
        try:
            module_ = importlib.import_module(module_name)
            try:
                class_ = getattr(module_, class_name)
            except AttributeError:
                raise AttributeError('Class does not exist')
        except ImportError:
            raise ImportError('Module does not exist')
        return class_

    @staticmethod
    def init_params_without_name(module_name, cfg):
        class_ = Configurator.str_to_class(module_name, cfg.name)
        init_dict = dict(cfg)
        del init_dict["name"]
        return class_(**init_dict)

    def _init_data_module(self):
        return self.init_params_without_name("src.data", self.cfg.data_module)

    def _init_sampler(self):
        return self.init_params_without_name("src.samplers", self.cfg.model.sampler)

    def _init_val_sampler(self):
        return self.init_params_without_name("src.samplers", self.cfg.model.val_sampler)

    def _init_sino_reconstructor(self):
        return self.init_params_without_name("src.reconstructors.sino", self.cfg.model.sino_reconstructor)

    def _init_image_reconstructor(self):
        return self.init_params_without_name("src.reconstructors.image", self.cfg.model.image_reconstructor)

    def _init_train_loss(self):
        return self.init_params_without_name("src.losses", self.cfg.train_loss)

    def _init_val_test_loss(self):
        return self.init_params_without_name("src.losses", self.cfg.val_test_loss)

    def _init_radon_transform(self):
        return self.init_params_without_name("src.radon_transforms", self.cfg.radon_transform)

    def _init_exp(self):
        if self.cfg.procedure.name == 'TrainVal':
            exp = TrainVal(self.cfg)
        else:
            raise NotImplementedError(
                f"Procedure {self.cfg.procedure.name} is not implemented.")

        return exp

    def _init_ckpt(self, exp_dir):
        # Allow pointing directly at a .ckpt file (e.g. a downloaded checkpoint).
        p = pathlib.Path(exp_dir)
        if p.is_file() and p.suffix == ".ckpt":
            print("Successfully loaded checkpoint from", p.name)
            return p

        all_ckpts = [c for c in p.iterdir() if c.suffix == ".ckpt"]
        if not all_ckpts:
            raise FileNotFoundError(f"No checkpoint files in {exp_dir!r}")

        # Prefer training-output checkpoints named "best*.ckpt"; otherwise fall
        # back to any .ckpt in the dir (e.g. pretrained weights downloaded from
        # the Hub, named cto_aapm.ckpt / cto_kits.ckpt).
        best_ckpts = [c for c in all_ckpts if c.stem.startswith("best")]
        ckpt_paths = best_ckpts if best_ckpts else all_ckpts

        def version(c: pathlib.Path) -> int:
            # match "best-vN" or just "best"
            m = re.fullmatch(r"best(?:-v(\d+))?", c.stem)
            return int(m.group(1)) if m and m.group(1) else 0

        latest = max(ckpt_paths, key=version)
        if len(ckpt_paths) > 1:
            print(
                f"Warning: multiple checkpoints found, using {latest.name!r}")
        print("Successfully loaded checkpoint from", latest.name)
        return latest

    def _load_model_from_ckpt(self, ckpt, model):
        # optional: add weights_only=True if on PyTorch >=2.4
        with safe_globals([CfgNode]):
            ckpt_obj = torch.load(ckpt, map_location='cpu', weights_only=True)
        state = ckpt_obj['state_dict']

        # If no init_modules in cfg, load everything (lenient)
        mods = getattr(self.cfg, 'init_modules', None)
        if mods is None:
            model.load_state_dict(state, strict=False)
            print(
                f"Loaded full checkpoint from {ckpt} (strict=False; no init_modules in cfg).")
            return model

        # Otherwise keep your filtered init logic
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in model_dict and k.split('.')[0] in mods}
        filtered_mods = set(k.split('.')[0] for k in filtered.keys())
        model_dict.update(filtered)
        model.load_state_dict(model_dict)

        # Trainability only if both lists exist
        train_flags = getattr(self.cfg, 'init_module_trainability', [])
        for m, t in zip(mods, train_flags):
            for p in getattr(model, m).parameters():
                p.requires_grad = t

        print(
            f"Model initialized from {ckpt} with modules {mods} and trainability {train_flags}")
        print(f"Actual filtered modules: {filtered_mods}")
        return model

    def _init_model(self):
        sampler = self._init_sampler()
        val_sampler = self._init_val_sampler()
        train_loss = self._init_train_loss()
        val_test_loss = self._init_val_test_loss()

        if self.cfg.procedure.name == 'TrainVal':
            sino_reconstructor = self._init_sino_reconstructor()
            image_reconstructor = self._init_image_reconstructor()

            model = VarnetLightningModule(
                    cfg=self.cfg,
                    sampler=sampler,
                    sino_reconstructor=sino_reconstructor,
                    image_reconstructor=image_reconstructor,
                    val_sampler = val_sampler,
                    train_loss=train_loss,
                    val_test_loss=val_test_loss
                )

        return model

    # def _init_radon_transform_kwargs(self):
    #     if self.cfg.radon_transform.name is None:
    #         raise ValueError("Radon transform kwargs name must be specified in the config.")

    #     rt_name = self.cfg.radon_transform.name

    #     rt_kwargs = RadonTransformKwargs().__dict__[rt_name]

    #     return CN(rt_kwargs)


    def init_all(self):

        model = self._init_model()

        if 'init_exp_dir' in self.cfg.keys():
            ckpt = self._init_ckpt(self.cfg.init_exp_dir)
            model = self._load_model_from_ckpt(ckpt, model)

        exp = self._init_exp()

        data_module = self._init_data_module()
        model.radon_transform = self._init_radon_transform()
        return exp, model, data_module


class TestConfigurator(Configurator):
    def __init__(self, args):
        """
        Create configs and make fixes
        """
        super().__init__(args)

    def init_all(self):
        model = self._init_model()

        if 'init_exp_dir' in self.cfg.keys():
            ckpt = self._init_ckpt(self.cfg.init_exp_dir)
            model = self._load_model_from_ckpt(ckpt, model)
        else:
            ckpt = self._init_ckpt(self.cfg.exp_dir)
            model = self._load_model_from_ckpt(ckpt, model)

        exp = Test(self.cfg)

        data_module = self._init_data_module()
        model.radon_transform = self._init_radon_transform()
        return exp, model, data_module
