import time
import logging
from tqdm import tqdm
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

from GANBandit.model import *
from GANBandit.data import Rec, Sync, Bibtext
from GANBandit.draw import *
from GANBandit.utils import configure_logging, set_seed


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('dataset', type=str,
                    help='an integer for the accumulator')
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size of bandit in one round')
parser.add_argument('--neural_bs', default=64, type=int,
                    help='batch size of neural network data')
parser.add_argument('--n_class', default=100, type=int,
                    help='n actions')
parser.add_argument('--n_sample', default=1000, type=int,
                    help='n data instance')
parser.add_argument('--n_dim', default=2, type=int,
                    help='feature dimension')
parser.add_argument('--seed', default=87, type=int,
                    help='random seed')
parser.add_argument('--checkpoint', default="checkpoint", type=str,
                    help='save model dir')
parser.add_argument('--output', default="output", type=str,
                    help='output dir')


args = parser.parse_args()
Path(args.checkpoint).mkdir(parents=True, exist_ok=True)
Path(args.output).mkdir(parents=True, exist_ok=True)


# rounds are simulated from the full dataset
def simulate_rounds(model, rewards, actions_hist, X_global, y_global, batch_st, batch_end):
    np.random.seed(batch_st)
    
    ## choosing actions for this batch
    actions_this_batch = model.predict(X_global[batch_st:batch_end, :]).astype('uint8')
    
    # keeping track of the sum of rewards received
    rewards.append(y_global[np.arange(batch_st, batch_end), actions_this_batch].sum())
    
    # adding this batch to the history of selected actions
    new_actions_hist = np.append(actions_hist, actions_this_batch)
    
    # now refitting the algorithms after observing these new rewards
    set_seed(batch_st)
    
    # print(X_global.shape, len(new_actions_hist), y_global.shape)
    
    model.fit(X_global[:batch_end, :], new_actions_hist, y_global[np.arange(batch_end), new_actions_hist])
    
    return new_actions_hist


def main():
    if args.dataset.casefold() == "sync":
        dataset = Sync(args.n_sample, args.n_dim, args.n_class)
    elif args.dataset.casefold() == "rec":
        dataset = Rec("../../ml-latest-small/ratings.csv")
    elif args.dataset.casefold() == "bibtext":
        dataset = Bibtext("../../Bibtex/Bibtex_data.txt")
    else:
        raise Exception("unknown dataset")
        
    timestamp = time.time()
    X, y = dataset.X, dataset.y
    nchoices = y.shape[1]
    batch_size = args.batch_size
    models = [MultiClass(nchoices=9724, epsilon=0.2, name="mul_lin_eg", seed=args.seed)]
    for model in models:
        model.X = X
        model.y = y

    # initial seed - all policies start with the same small random selection of actions/rewards
    set_seed(args.seed)
    
    logging.info("Prepare first batch.")
    first_batch = X[:batch_size, :]
    nchoices = y.shape[1]
    action_chosen = np.random.randint(nchoices, size=batch_size)
    rewards_received = y[np.arange(batch_size), action_chosen]
    
    logging.info("Fitting models for the first time.")
    for model in models:
        if not hasattr(model, "lst_action"):
            model.fit(X=first_batch, a=action_chosen, r=rewards_received)
            
    lst_actions = [action_chosen.copy() for i in models]
    lst_rewards = [[] for i in models]
            
    n_simulation = 3  # int(np.floor(X.shape[0] / batch_size))
    logging.info(f"Running all {n_simulation} simulation")
    for i in tqdm(range(n_simulation)):
        batch_st = (i + 1) * batch_size
        batch_end = (i + 2) * batch_size
        batch_end = np.min([batch_end, X.shape[0]])

        for e, model in enumerate(models):
            try:
                lst_actions[e] = simulate_rounds(models[e],
                                                 lst_rewards[e],
                                                 lst_actions[e],
                                                 X, y,
                                                 batch_st, batch_end)
                models[e].save(i)
                #models[model].load(i)
                model.lst_actions = lst_actions[e]
                model.lst_rewards = lst_rewards[e]
            except:
                print(model)
                raise
                
    draw_cum_reward(models, batch_size, args.dataset, y.shape[1], X.shape[1], args.output, timestamp)
    draw_train_loss(models, args.output, timestamp)


if __name__ == "__main__":
    configure_logging()
    main()