import argparse
import time
import datetime
import tensorboardX
import sys
import numpy as np
import algo_vae
import rl_utils
from rl_utils import device
from model_policy import ACModel
import pickle
from model_f import RepresentationModel


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
    parser.add_argument("--procs", type=int, default=16,
                        help="number of processes (default: 16)")
    parser.add_argument("--frames", type=int, default=10**7,
                        help="number of frames of training (default: 1e7)")

    ## Parameters for main algorithm
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--epochs_g", type=int, default=16,
                        help="number of epochs for training z")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for PPO (default: 256)")
    parser.add_argument("--batch-size-g", type=int, default=1024,
                        help="batch size (default: 1024)")
    parser.add_argument("--frames_per_proc", type=int, default=None,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--lr-g", type=float, default=0.0003,
                        help="learning rate (default: 0.0003)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). \
                        If > 1, a LSTM is added to the model to have memory.\
                        If policy is memoryless, then set recurrence = 1")
    parser.add_argument("--history-recurrence", type=int, default=8,
                        help="number of time-steps gradient is backpropagated (default: 1).")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")
    parser.add_argument("--random-action", action="store_true", default=False,
                        help="Whether to collect experiences using a random policy")
    parser.add_argument("--num-frames-g", type=int, default=0,
                        help="How many frames to pre-train g")
    parser.add_argument("--latent-dim-vae", type=int, default=8,
                        help="number of latent dimensions in VAE")
    parser.add_argument("--latent-dim-f", type=int, default=16,
                        help="Latent Dimension of representation learning model distribution parameters")
    parser.add_argument("--beta", type=float, default=0.0001,
                        help="KL loss parameter")
    parser.add_argument("--rep-model-dir", default=None,
                        help="Directory name of the representation learning model")

    args = parser.parse_args()

    # args.mem describes whether the POLICY uses memory
    if args.algo == "belief_vae":
        args.mem = False
    else:
        args.mem = args.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"train_{args.env}_{args.algo}_seed{args.seed}"

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

    # Load environments
    envs = []
    for i in range(args.procs):
        envs.append(rl_utils.make_env(args.env, args.seed + 10000 * i))
    txt_logger.info("Environments loaded\n")


    # Load training status
    try:
        status = rl_utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    obs_space, preprocess_obss = rl_utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    aux_info = False
    aux_info_embedding_size = 0

    acmodel = ACModel(obs_space, envs[0].action_space, aux_info_embedding_size, args.algo, args.mem, args.text,
                      aux_info, latent_dim_f=args.latent_dim_f, use_cnn='EscapeRoom' in args.env)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    rep_model = RepresentationModel(obs_space=obs_space, action_space=envs[0].action_space,
                                    latent_dim=args.latent_dim_f, use_cnn='EscapeRoom' in args.env).to(device)
    txt_logger.info("RepModel loaded\n")
    txt_logger.info("{}\n".format(rep_model))

    model_name = str(args.env) + "_representation_learning_seed" + str(args.seed)  # args.rep_model_dir
    rep_model_dir = rl_utils.get_model_dir(model_name)
    rep_model_checkpoint = rl_utils.get_status(rep_model_dir)
    rep_model.load_state_dict(rep_model_checkpoint["model_state"])

    # Load algo
    algo = algo_vae.Algo(envs, args.env, acmodel, args.algo, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                         args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                         args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss,
                         history_recurrence=args.history_recurrence, batch_size_g=args.batch_size_g, lr_g=args.lr_g,
                         epochs_g=args.epochs_g, rep_model=rep_model, latent_dim=args.latent_dim_vae, latent_dim_f=args.latent_dim_f,
                         tb_writer=tb_writer, beta=args.beta, seed=args.seed, use_cnn='Escape' in args.env)

    txt_logger.info("{}\n".format(algo.belief_vae))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    if "vae_model_state" in status:
        algo.belief_vae.load_state_dict(status["vae_model_state"])
    if "vae_optimizer_state" in status:
        algo.vae_optimizer.load_state_dict(status["vae_optimizer_state"])

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()


    while num_frames < args.frames:
        # Update model parameters

        update_start_time = time.time()
        if num_frames < args.num_frames_g:
            exps, logs, mylogs = algo.collect_experiences(random_action=True)
            logs1 = algo.update_g_parameters(exps)
            logs = {**logs, **logs1}
        else:
            print("collecting data")
            exps, logs = algo.collect_experiences(random_action=False)
            print("updating g params")
            logs1 = algo.update_g_parameters(exps)
            print("updating params")
            logs2 = algo.update_parameters(exps)
            logs = {**logs, **logs1, **logs2}

            # Save status

            if args.save_interval > 0 and update % args.save_interval == 0:
                status = {"num_frames": num_frames,
                          "update": update,
                          "model_state": algo.acmodel.state_dict(),
                          "optimizer_state": algo.optimizer.state_dict(),
                          "vae_model_state": algo.belief_vae.state_dict(),
                          "vae_optimizer_state": algo.vae_optimizer.state_dict(),
                          }
                rl_utils.save_status(status, model_dir)
                txt_logger.info("Status saved")


            update_end_time = time.time()
            # Print logs
            if update % args.log_interval == 0:

                with open(f'{model_dir}/epoch_{num_frames}.pkl', 'wb') as file:
                    pickle.dump(logs, file)

                fps = logs["num_frames"] / (update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                return_per_episode = rl_utils.synthesize(logs["return_per_episode"])
                rreturn_per_episode = rl_utils.synthesize(logs["reshaped_return_per_episode"])
                num_frames_per_episode = rl_utils.synthesize(logs["num_frames_per_episode"])

                header = ["update", "frames", "FPS", "duration"]
                data = [update, num_frames, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

                if args.algo == "belief_vae" and num_frames < args.num_frames_g:
                    header += ["batch_elbo_loss"]
                    # data += [logs["batch_elbo_loss"]]
                    data += [0]
                    txt_logger.info(
                        "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | elbo_loss {:.3f}"
                        .format(*data))
                else:
                    header += ["batch_elbo_loss", "grad_norm", "starting_elbo"]
                    data += [logs["batch_elbo_loss"], logs["grad_norm"], logs["starting_elbo"]]
                    # data += [0,0]
                    txt_logger.info(
                        "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | elbo_loss {:.3f} | grad_norm_g {:.3f} | starting_elbo {:.3f}"
                        .format(*data))

                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()

                if status["num_frames"] == 0:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()

                for field, value in zip(header, data):
                    tb_writer.add_scalar(field, value, num_frames)

        num_frames += logs["num_frames"]
        update += 1

