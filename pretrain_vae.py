import argparse
import time
import datetime
import tensorboardX
import sys
import algo_vae_pretrain
import rl_utils
from rl_utils import device
from model_f import RepresentationModel
import torch
from torch_ac.utils import DictList
import numpy

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    ## General parameters
    parser.add_argument("--algo", required=True,
                        help="algorithm to use: ppo |  belief_vae (REQUIRED)")
    parser.add_argument("--env", required=True,
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--data-path", default=None,
                        help="Path of collected data")

    ## Parameters for main algorithm
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--epochs_g", type=int, default=16,
                        help="number of epochs for training z")
    parser.add_argument("--batch-size-g", type=int, default=1024,
                        help="batch size (default: 1024)")
    parser.add_argument("--lr-g", type=float, default=0.0003,
                        help="learning rate (default: 0.0003)")
    parser.add_argument("--num-frames-g", type=int, default=0,
                        help="How many frames to pre-train g")
    parser.add_argument("--latent-dim-vae", type=int, default=32,
                        help="number of latent dimensions in VAE")
    parser.add_argument("--latent-dim-f", type=int, default=16,
                        help="Latent Dimension of representation learning model distribution parameters")
    parser.add_argument("--beta", type=float, default=0.0001,
                        help="KL loss parameter")
    parser.add_argument("--rep-model-dir", default=None,
                        help="Directory name of the representation learning model")

    args = parser.parse_args()

    # args.mem describes whether the POLICY uses memory
    args.mem = False

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}"

    model_name = args.model or default_model_name
    model_dir = rl_utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer
    txt_logger = rl_utils.get_txt_logger(model_dir)
    csv_file, csv_logger = rl_utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources
    rl_utils.seed(args.seed)

    # Set device
    txt_logger.info(f"Device: {device}\n")

    # Load training status
    try:
        status = rl_utils.get_status(model_dir)
    except OSError:
        status = {"epochs": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    env = rl_utils.make_env(args.env, args.seed)
    obs_space, preprocess_obss = rl_utils.get_obss_preprocessor(env.observation_space)
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    rep_model = RepresentationModel(obs_space=obs_space, action_space=env.action_space,
                                    latent_dim=args.latent_dim_f, use_cnn='Escape' in args.env).to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(rep_model))

    model_name = str(args.env) + "_representation_learning_seed" + str(args.seed)
    rep_model_dir = rl_utils.get_model_dir(model_name)
    rep_model_checkpoint = rl_utils.get_status(rep_model_dir)
    rep_model.load_state_dict(rep_model_checkpoint["model_state"])
    

    # Load algo
    algo = algo_vae_pretrain.Algo(env, device,
                         args.optim_eps, lr_g=args.lr_g,
                         epochs_g=args.epochs_g, rep_model=rep_model,
                         latent_dim=args.latent_dim_vae, latent_dim_f=args.latent_dim_f,
                         tb_writer=tb_writer, beta=args.beta, use_cnn='Escape' in args.env)

    txt_logger.info("{}\n".format(algo.belief_vae))

    if "vae_model_state" in status:
        algo.belief_vae.load_state_dict(status["vae_model_state"])
    if "vae_optimizer_state" in status:
        algo.vae_optimizer.load_state_dict(status["vae_optimizer_state"])

    # Train model

    epochs = status["epochs"]
    update = status["update"]
    start_time = time.time()


    data = torch.load(args.data_path)


    obs = data["obss"]
    state = data["states"]
    # exps.action = data["actions"].transpose(0,1).to(device)
    mask = data["masks"]
    reward = data["rewards"]
    if "Genie" in env.__class__.__name__:
        item_location = data["item_locations"]
    elif "ModifiedCookie" in env.__class__.__name__:
        cookie_locations = data["cookie_locations"]
        button_locations = data["button_locations"]
        room_belief = data["room_belief"]
        cookie_belief = data["cookie_belief"]

    while epochs < args.epochs_g:
        # Update model parameters
        update_start_time = time.time()

        indexes = numpy.arange(0, obs.shape[0])
        indexes = numpy.random.permutation(indexes)
        batch_size = args.batch_size_g
        indexes = indexes[:batch_size]
        exps = DictList()
        exps.obs = obs[indexes].transpose(0,1).to(device)
        exps.state = state[indexes].transpose(0, 1).to(device)
        exps.reward = reward[indexes].transpose(0, 1).to(device)
        if "Genie" in env.__class__.__name__:
            exps.item_location = item_location[indexes].transpose(0, 1).to(device)
        elif "ModifiedCookie" in env.__class__.__name__:
            exps.cookie_locations = cookie_locations[indexes].transpose(0, 1).to(device)
            exps.button_locations = button_locations[indexes].transpose(0, 1).to(device)
            exps.room_belief = room_belief[indexes].transpose(0, 1).to(device)
            exps.cookie_belief = cookie_belief[indexes].transpose(0, 1).to(device)
        exps.mask = mask[indexes].transpose(0, 1).to(device)

        logs = algo.update_g_parameters(exps)
        update_end_time = time.time()

        # Save status
        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"epochs": epochs,
                      "update": update,
                      "vae_model_state": algo.belief_vae.state_dict(),
                      "vae_optimizer_state": algo.vae_optimizer.state_dict(),
                      }
            rl_utils.save_status(status, model_dir)
            txt_logger.info("Status saved")

        # Print logs
        if update % args.log_interval == 0:
            header = ["batch_elbo_loss"]
            data = [logs["batch_elbo_loss"]]
            txt_logger.info("elbo_loss {:.3f}".format(*data))

            if status["epochs"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

        print(epochs, "epochs")
        epochs += 1
        update += 1

