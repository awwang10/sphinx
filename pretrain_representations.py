import argparse
import tensorboardX
import sys
import rl_utils
from rl_utils import device
from model_f import RepresentationModel
import algo_f
import torch
from torch_ac.utils import DictList
import torch.nn.functional as F

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    ## General parameters
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
    parser.add_argument("--epochs", type=int, default=20000,
                        help="number of epochs for training z")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="batch size (default: 1024)")
    parser.add_argument("--lr", type=float, default=0.0003,
                        help="learning rate (default: 0.0003)")
    parser.add_argument("--latent-dim", type=int, default=16,
                        help="Latent Dimension of representation learning model distribution parameters")
    parser.add_argument("--dynamics-loss-s-coef", type=float, default=0.1,
                        help="dynamics loss parameter")
    parser.add_argument("--dynamics-loss-o-coef", type=float, default=0.1,
                        help="dynamics loss parameter")
    parser.add_argument("--reward-loss-coef", type=float, default=0.1,
                        help="reward loss parameter")    
    parser.add_argument("--beta", type=float, default=0.0001,
                        help="KL loss parameter")

    args = parser.parse_args()

    # Set run dir

    default_model_name = f"{args.env}_representation_learning_seed{args.seed}"
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

    # Load env
    env = rl_utils.make_env_with_env_name(args.env, args_procs=1, seed=args.seed)[0]
    obs_space, _ = rl_utils.get_obss_preprocessor(env.observation_space)
    txt_logger.info("Environments loaded\n")

    # Load representation model
    rep_model = RepresentationModel(obs_space=obs_space, action_space=env.action_space, use_cnn='EscapeRoom' in args.env, latent_dim=args.latent_dim).to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(rep_model))
    if "model_state" in status:
        rep_model.load_state_dict(status["model_state"])

    # Load data
    data = torch.load(args.data_path)

    eps_len = data["masks"].shape[1]
    indices = (data["masks"] == 1).nonzero(as_tuple=True)
    next_indices = (indices[0], indices[1]+1)
    next_indices_clamped = (indices[0], torch.clamp(indices[1]+1, max=eps_len-1))

    exps = DictList()

    exps.obs = DictList({'image': data["obss"][indices].to(device)})
    exps.state = DictList({'image': data["states"][indices].to(device)})
    exps.next_obs = DictList({'image': data["obss"][next_indices_clamped].to(device)})
    exps.next_state = DictList({'image': data["states"][next_indices_clamped].to(device)})

    exps.next_mask = F.pad(input=data["masks"], pad=(0, 1))[next_indices].to(device)
    exps.action = data["actions"][indices].to(device)
    exps.reward = data["rewards"][indices].to(device)

    # Load algo
    algo = algo_f.Algo(env, exps, rep_model, device, args.optim_eps, batch_size=args.batch_size, lr=args.lr, tb_writer=tb_writer,
                        beta=args.beta, dynamics_loss_s_coef=args.dynamics_loss_s_coef, dynamics_loss_o_coef=args.dynamics_loss_o_coef, reward_loss_coef = args.reward_loss_coef)

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])


    # Train model
    epochs = status["epochs"]
    update = status["update"]

    while epochs < args.epochs:
        # Update model parameters

        logs = algo.update_f_parameters()

        # Save status
        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"epochs": epochs,
                      "update": update,
                      "model_state": algo.rep_model.state_dict(),
                      "optimizer_state": algo.optimizer.state_dict(),
                      }
            rl_utils.save_status(status, model_dir)
            txt_logger.info("Status saved")

        # Print logs
        if update % args.log_interval == 0:
            header = ["state_dynamics_loss", "obs_dynamics_loss", "reward_loss", "kl_loss", "grad_norm"]
            data = [logs["state_dynamics_loss"], logs["obs_dynamics_loss"], logs["reward_loss"], logs["kl_loss"], logs["grad_norm"]]
            txt_logger.info("s-loss: {:.3f}, o-loss: {:.3f}, r-loss: {:.3f}, kl-loss: {:.3f}, ∇: {:.3f}".format(*data))


            if status["epochs"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()


        print(epochs, "epochs")
        epochs += 1
        update += 1

