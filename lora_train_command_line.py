import time
from typing import Optional, Union
import os
import json

import train_network
import library.train_util as util
import argparse


class ArgStore:
    # Represents the entirety of all possible inputs for sd-scripts. they are ordered from most important to least
    def __init__(self):
        # 重要部分，多数情况下需要修改
        self.base_model: str = r"/home/zhe/stable-diffusion-webui/sd-scripts/model/orangechillmix_v70.safetensors"  # 底模(base model)位置
        self.img_folder: str = r"/home/zhe/stable-diffusion-webui/sd-scripts/train_data/images"  # 预处理好的图片文件夹所在位置，其文件目录结构参考: https://rentry.org/2chAI_LoRA_Dreambooth_guide_english#for-kohyas-script
        self.output_folder: str = r"/home/zhe/stable-diffusion-webui/sd-scripts/output_models"  # 训练中所有模型输出文件夹
        self.change_output_name: Union[str, None] = None  # changes the output name of the epochs
        self.save_json_folder: Union[str, None] = None  # OPTIONAL, saves a json folder of your config to whatever location you set here.
        self.load_json_path: Union[str, None] = None  # OPTIONAL, loads a json file partially changes the config to match. things like folder paths do not get modified.
        self.json_load_skip_list: Union[list[str], None] = None  # OPTIONAL, allows the user to define what they skip when loading a json, by default it loads everything, including all paths, set it up like this ["base_model", "img_folder", "output_folder"]

        self.net_dim: int = 128  # network dimension, 128 is the most common, however you might be able to get lesser to work，并不是越大越好,缩小该选项可以少量减少显存占用
        self.alpha: float = 64  # represents the scalar for training. the lower the alpha, the less gets learned per step. if you want the older way of training, set this to dim
        # list of schedulers: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup
        self.scheduler: str = "cosine_with_restarts"  # the scheduler for learning rate. Each does something specific
        self.cosine_restarts: Union[int, None] = 1  # OPTIONAL, represents the number of times it restarts. Only matters if you are using cosine_with_restarts
        self.scheduler_power: Union[float, None] = 1  # OPTIONAL, represents the power of the polynomial. Only matters if you are using polynomial
        self.warmup_lr_ratio: Union[float, None] = None  # OPTIONAL, Calculates the number of warmup steps based on the ratio given. Make sure to set this if you are using constant_with_warmup, None to ignore
        self.learning_rate: Union[float, None] = 1e-4  # OPTIONAL, when not set, lr gets set to 1e-3 as per adamW. Personally, I suggest actually setting this as lower lr seems to be a small bit better.
        self.text_encoder_lr: Union[float, None] = None  # OPTIONAL, Sets a specific lr for the text encoder, this overwrites the base lr I believe, None to ignore
        self.unet_lr: Union[float, None] = None  # OPTIONAL, Sets a specific lr for the unet, this overwrites the base lr I believe, None to ignore
        self.num_workers: int = 6  # The number of threads that are being used to load images, lower speeds up the start of epochs, but slows down the loading of data. The assumption here is that it increases the training time as you reduce this value

        self.batch_size: int = 1  # 单次处理图片数，显著提高训练速度，但会占用大量显存，具体取决于分辨率和显存大小以及是否应用优化(xformers 8bit adma), this is directly proportional to your vram and resolution. with 12gb of vram, at 512 reso, you can get a maximum of 6 batch size
        self.num_epochs: int = 10  # epoch 数, 根据epoch计算step数，如果设定了step，该设置将失效
        self.save_at_n_epochs: Union[int, None] = 4 # OPTIONAL, 训练中保存模型的间隔，这有助于训练模型过拟合时回退，建议配合下面的继续训练选项进行调整， None to ignore
        self.shuffle_captions: bool = False  # OPTIONAL, False to ignore
        self.keep_tokens: Union[int, None] = None  # OPTIONAL, None to ignore
        self.max_steps: Union[int, None] = None  # OPTIONAL, 最大step数，直接制定则不会根据epoch进行计算. None to ignore

        # These are the second most likely things you will modify
        self.train_resolution: int = 512
        self.min_bucket_resolution: int = 320
        self.max_bucket_resolution: int = 1024
        self.lora_model_for_resume: Union[str, None] = None #'./output_models/last.safetensors'  # OPTIONAL, 从一个已有的lora模型开始训练，而不是从头开始，None 表示忽略
        self.save_state: bool = True  # OPTIONAL, 保存训练状态，用于中断后的继续训练 False 为不保存,将会占用大量硬盘，每一个state都包含一个base model!!!!!!!!!!!!输出位置:output_folder
        self.load_previous_save_state: Union[str, None] =None # './output_models/last-state' # "/home/zhe/stable-diffusion-webui/sd-scripts/models_state"  # OPTIONAL, 保存的训练状态文件路径，用于继续训练, None to ignore
        self.training_comment: Union[str, None] = None  # OPTIONAL, great way to put in things like activation tokens right into the metadata. seems to not work at this point and time

        # These are the least likely things you will modify
        self.reg_img_folder: Union[str, None] = None  # OPTIONAL, None to ignore
        self.clip_skip: int = 2  # If you are training on a model that is anime based, keep this at 2 as most models are designed for that
        self.test_seed: int = 23  # this is the "reproducable seed", 当你设定了该种子，则可以在生成是使用训练照片的提示词区再现该照片
        self.prior_loss_weight: float = 1  # is the loss weight much like Dreambooth, is required for LoRA training
        self.gradient_checkpointing: bool = False  # OPTIONAL, enables gradient checkpointing
        self.gradient_acc_steps: Union[int, None] = None  # OPTIONAL, not sure exactly what this means
        self.mixed_precision: str = "fp16"  # bf16 相比 fp16 牺牲了精度，换来了更大的范围，这可能对训练有好处
        self.save_precision: str = "fp16"  # You can also save in bf16, but because it's not universally supported, I suggest you keep saving at fp16
        self.save_as: str = "safetensors"  # list is pt, ckpt, safetensors
        self.caption_extension: str = ".txt"  # the other option is .captions, but since wd1.4 tagger outputs as txt files, this is the default
        self.max_clip_token_length = 150  # can be 75, 150, or 225 I believe, there is no reason to go higher than 150 though
        self.buckets: bool = True
        self.xformers: bool = False       # AMD 显卡不可用
        self.use_8bit_adam: bool = False  # AMD 显卡不可用
        self.cache_latents: bool = True
        self.color_aug: bool = False  # IMPORTANT: Clashes with cache_latents, only have one of the two on!
        self.flip_aug: bool = False
        self.vae: Union[str, None] = None  # Seems to only make results worse when not using that specific vae, should probably not use
        self.no_meta: bool = False  # This removes the metadata that now gets saved into safetensors, (you should keep this on)
        self.log_dir: Union[str, None] = '/home/zhe/stable-diffusion-webui/sd-scripts/log'  # output of logs, not useful to most people.

    # Creates the dict that is used for the rest of the code, to facilitate easier json saving and loading
    @staticmethod
    def convert_args_to_dict():
        return ArgStore().__dict__


def main():
    parser = argparse.ArgumentParser()
    setup_args(parser)
    arg_dict = ArgStore.convert_args_to_dict()
    pre_args = parser.parse_args()
    if pre_args.load_json_path or arg_dict["load_json_path"]:
        load_json(pre_args.load_json_path if pre_args.load_json_path else arg_dict['load_json_path'], arg_dict)
    if pre_args.save_json_path or arg_dict["save_json_folder"]:
        save_json(pre_args.save_json_path if pre_args.save_json_path else arg_dict['save_json_folder'], arg_dict)
    args = create_arg_space(arg_dict)
    args = parser.parse_args(args)
    train_network.train(args)


def create_arg_space(args: dict) -> list[str]:
    if not ensure_path(args["base_model"], "base_model", {"ckpt", "safetensors"}):
        raise FileNotFoundError("Failed to find base model, make sure you have the correct path")
    if not ensure_path(args["img_folder"], "img_folder"):
        raise FileNotFoundError("Failed to find the image folder, make sure you have the correct path")
    if not ensure_path(args["output_folder"], "output_folder"):
        raise FileNotFoundError("Failed to find the output folder, make sure you have the correct path")
    # This is the list of args that are to be used regardless of setup
    output = ["--network_module=networks.lora", f"--pretrained_model_name_or_path={args['base_model']}",
              f"--train_data_dir={args['img_folder']}", f"--output_dir={args['output_folder']}",
              f"--prior_loss_weight={args['prior_loss_weight']}", f"--caption_extension=" + args['caption_extension'],
              f"--resolution={args['train_resolution']}", f"--train_batch_size={args['batch_size']}",
              f"--mixed_precision={args['mixed_precision']}", f"--save_precision={args['save_precision']}",
              f"--network_dim={args['net_dim']}", f"--save_model_as={args['save_as']}",
              f"--clip_skip={args['clip_skip']}", f"--seed={args['test_seed']}",
              f"--max_token_length={args['max_clip_token_length']}", f"--lr_scheduler={args['scheduler']}",
              f"--network_alpha={args['alpha']}"]
    if not args['max_steps']:
        steps = find_max_steps(args)
    else:
        steps = args["max_steps"]
    output.append(f"--max_train_steps={steps}")
    output += create_optional_args(args, steps)
    return output


def create_optional_args(args: dict, steps):
    output = []
    if args["reg_img_folder"]:
        if not ensure_path(args["reg_img_folder"], "reg_img_folder"):
            raise FileNotFoundError("Failed to find the reg image folder, make sure you have the correct path")
        output.append(f"--reg_data_dir={args['reg_img_folder']}")

    if args['lora_model_for_resume']:
        if not ensure_path(args['lora_model_for_resume'], "lora_model_for_resume", {"pt", "ckpt", "safetensors"}):
            raise FileNotFoundError("Failed to find the lora model, make sure you have the correct path")
        output.append(f"--network_weights={args['lora_model_for_resume']}")

    if args['save_at_n_epochs']:
        output.append(f"--save_every_n_epochs={args['save_at_n_epochs']}")
    else:
        output.append("--save_every_n_epochs=999999")

    if args['shuffle_captions']:
        output.append("--shuffle_caption")

    if args['keep_tokens'] and args['keep_tokens'] > 0:
        output.append(f"--keep_tokens={args['keep_tokens']}")

    if args['buckets']:
        output.append("--enable_bucket")
        output.append(f"--min_bucket_reso={args['min_bucket_resolution']}")
        output.append(f"--max_bucket_reso={args['max_bucket_resolution']}")

    if args['use_8bit_adam']:
        output.append("--use_8bit_adam")

    if args['xformers']:
        output.append("--xformers")

    if args['color_aug']:
        if args['cache_latents']:
            print("color_aug and cache_latents conflict with one another. Please select only one")
            quit(1)
        output.append("--color_aug")

    if args['flip_aug']:
        output.append("--flip_aug")

    if args['cache_latents']:
        output.append("--cache_latents")

    if args['warmup_lr_ratio'] and args['warmup_lr_ratio'] > 0:
        warmup_steps = int(steps * args['warmup_lr_ratio'])
        output.append(f"--lr_warmup_steps={warmup_steps}")

    if args['gradient_checkpointing']:
        output.append("--gradient_checkpointing")

    if args['gradient_acc_steps'] and args['gradient_acc_steps'] > 0 and args['gradient_checkpointing']:
        output.append(f"--gradient_accumulation_steps={args['gradient_acc_steps']}")

    if args['learning_rate'] and args['learning_rate'] > 0:
        output.append(f"--learning_rate={args['learning_rate']}")

    if args['text_encoder_lr'] and args['text_encoder_lr'] > 0:
        output.append(f"--text_encoder_lr={args['text_encoder_lr']}")

    if args['unet_lr'] and args['unet_lr'] > 0:
        output.append(f"--unet_lr={args['unet_lr']}")

    if args['vae']:
        output.append(f"--vae={args['vae']}")

    if args['no_meta']:
        output.append("--no_metadata")

    if args['save_state']:
        output.append("--save_state")

    if args['load_previous_save_state']:
        if not ensure_path(args['load_previous_save_state'], "previous_state"):
            raise FileNotFoundError("Failed to find the save state folder, make sure you have the correct path")
        output.append(f"--resume={args['load_previous_save_state']}")

    if args['change_output_name']:
        output.append(f"--output_name={args['change_output_name']}")

    if args['training_comment']:
        output.append(f"--training_comment={args['training_comment']}")

    if args['num_workers']:
        output.append(f"--max_data_loader_n_workers={args['num_workers']}")

    if args['cosine_restarts'] and args['scheduler'] == "cosine_with_restarts":
        output.append(f"--lr_scheduler_num_cycles={args['cosine_restarts']}")

    if args['scheduler_power'] and args['scheduler'] == "polynomial":
        output.append(f"--lr_scheduler_power={args['scheduler_power']}")
    return output


def find_max_steps(args: dict) -> int:
    total_steps = 0
    folders = os.listdir(args["img_folder"])
    for folder in folders:
        if not os.path.isdir(os.path.join(args["img_folder"], folder)):
            continue
        num_repeats = folder.split("_")
        if len(num_repeats) < 2:
            print(f"folder {folder} is not in the correct format. Format is x_name. skipping")
            continue
        try:
            num_repeats = int(num_repeats[0])
        except ValueError:
            print(f"folder {folder} is not in the correct format. Format is x_name. skipping")
            continue
        imgs = 0
        for file in os.listdir(os.path.join(args["img_folder"], folder)):
            if os.path.isdir(file):
                continue
            ext = file.split(".")
            if ext[-1].lower() in {"png", "bmp", "gif", "jpeg", "jpg", "webp"}:
                imgs += 1
        total_steps += (num_repeats * imgs)
    total_steps = int((total_steps / args["batch_size"]) * args["num_epochs"])
    if quick_calc_regs(args):
        total_steps *= 2
    return total_steps


def quick_calc_regs(args: dict) -> bool:
    if not args["reg_img_folder"] or not os.path.exists(args["reg_img_folder"]):
        return False
    folders = os.listdir(args["reg_img_folder"])
    for folder in folders:
        if not os.path.isdir(os.path.join(args["reg_img_folder"], folder)):
            continue
        num_repeats = folder.split("_")
        if len(num_repeats) < 2:
            continue
        try:
            num_repeats = int(num_repeats[0])
        except ValueError:
            continue
        for file in os.listdir(os.path.join(args["reg_img_folder"], folder)):
            if os.path.isdir(file):
                continue
            ext = file.split(".")
            if ext[-1].lower() in {"png", "bmp", "gif", "jpeg", "jpg", "webp"}:
                return True
    return False


def add_misc_args(parser) -> None:
    parser.add_argument("--save_json_path", type=str, default=None,
                        help="Path to save a configuration json file to")
    parser.add_argument("--load_json_path", type=str, default=None,
                        help="Path to a json file to configure things from")
    parser.add_argument("--no_metadata", action='store_true',
                        help="do not save metadata in output model / メタデータを出力先モデルに保存しない")
    parser.add_argument("--save_model_as", type=str, default="safetensors", choices=[None, "ckpt", "pt", "safetensors"],
                        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）")

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None,
                        help="learning rate for Text Encoder / Text Encoderの学習率")
    parser.add_argument("--lr_scheduler_num_cycles", type=int, default=1,
                        help="Number of restarts for cosine scheduler with restarts / cosine with restartsスケジューラでのリスタート回数")
    parser.add_argument("--lr_scheduler_power", type=float, default=1,
                        help="Polynomial power for polynomial scheduler / polynomialスケジューラでのpolynomial power")

    parser.add_argument("--network_weights", type=str, default=None,
                        help="pretrained weights for network / 学習するネットワークの初期重み")
    parser.add_argument("--network_module", type=str, default=None,
                        help='network module to train / 学習対象のネットワークのモジュール')
    parser.add_argument("--network_dim", type=int, default=None,
                        help='network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）')
    parser.add_argument("--network_alpha", type=float, default=1,
                        help='alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）')
    parser.add_argument("--network_args", type=str, default=None, nargs='*',
                        help='additional argmuments for network (key=value) / ネットワークへの追加の引数')
    parser.add_argument("--network_train_unet_only", action="store_true",
                        help="only training U-Net part / U-Net関連部分のみ学習する")
    parser.add_argument("--network_train_text_encoder_only", action="store_true",
                        help="only training Text Encoder part / Text Encoder関連部分のみ学習する")
    parser.add_argument("--training_comment", type=str, default=None,
                        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列")


def setup_args(parser) -> None:
    util.add_sd_models_arguments(parser)
    util.add_dataset_arguments(parser, True, True)
    util.add_training_arguments(parser, True)
    add_misc_args(parser)


def ensure_path(path: str, name: str, ext_list: Optional[list[str]] = None) -> bool:
    if ext_list is None:
        ext_list = []
    folder = len(ext_list) == 0
    if path is None or not os.path.exists(path):
        print(f"未找到{name}，请确保路径正确。")
        return False
    elif folder and os.path.isfile(path):
        print(f"{name}的路径是一个文件，请选择文件夹。")
        return False
    elif not folder and os.path.isdir(path):
        print(f"{name}的路径是一个文件夹，请选择文件。")
        return False
    elif not folder and path.split(".")[-1] not in ext_list:
        print(f"找到了一个{name}的文件，但它不是所接受的类型之一：{ext_list}")
        return False
    return True

def save_json(path: str, obj: dict) -> None:
    """
    将对象保存为JSON文件。

    参数：
    path：str，保存JSON文件的路径。
    obj：字典，要保存的对象。
    """
    ensure_path(path, "save_json_path")
    fp = open(os.path.join(path, f"config-{time.time()}.json"), "w")
    json.dump(obj, fp=fp, indent=4)
    fp.close()


def load_json(path, obj: dict) -> dict:
    if not ensure_path(path, "load_json_path", {"json"}):
        raise FileNotFoundError("Failed to find base model, make sure you have the correct path")
    with open(path) as f:
        json_obj = json.loads(f.read())
    print("loaded json, setting variables...")
    ui_name_scheme = {"pretrained_model_name_or_path": "base_model", "logging_dir": "log_dir",
                      "train_data_dir": "img_folder", "reg_data_dir": "reg_img_folder",
                      "output_dir": "output_folder", "max_resolution": "train_resolution",
                      "lr_scheduler": "scheduler", "lr_warmup": "warmup_lr_ratio",
                      "train_batch_size": "batch_size", "epoch": "num_epochs",
                      "save_every_n_epochs": "save_at_n_epochs", "num_cpu_threads_per_process": "num_workers",
                      "enable_bucket": "buckets", "save_model_as": "save_as", "shuffle_caption": "shuffle_captions",
                      "resume": "load_previous_save_state", "network_dim": "net_dim",
                      "gradient_accumulation_steps": "gradient_acc_steps", "output_name": "change_output_name",
                      "network_alpha": "alpha", "lr_scheduler_num_cycles": "cosine_restarts",
                      "lr_scheduler_power": "scheduler_power"}

    for key in list(json_obj):
        if key in ui_name_scheme:
            json_obj[ui_name_scheme[key]] = json_obj[key]
            if ui_name_scheme[key] in {"batch_size", "num_epochs"}:
                try:
                    json_obj[ui_name_scheme[key]] = int(json_obj[ui_name_scheme[key]])
                except ValueError:
                    print(f"attempting to load {key} from json failed as input isn't an integer")
                    quit(1)

    for key in list(json_obj):
        if obj["json_load_skip_list"] and key in obj["json_load_skip_list"]:
            continue
        if key in obj:
            if key in {"keep_tokens", "warmup_lr_ratio"}:
                json_obj[key] = int(json_obj[key]) if json_obj[key] is not None else None
            if key in {"learning_rate", "unet_lr", "text_encoder_lr"}:
                json_obj[key] = float(json_obj[key]) if json_obj[key] is not None else None
            if obj[key] != json_obj[key]:
                print_change(key, obj[key], json_obj[key])
                obj[key] = json_obj[key]
    print("completed changing variables.")
    return obj


def print_change(value, old, new):
    print(f"{value} changed from {old} to {new}")


if __name__ == "__main__":
    main()
