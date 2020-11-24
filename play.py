import time
import random
import torch
import numpy as np


def make_env(world, stage):
    from env import MarioEnv
    return MarioEnv(world, stage)


def make_model(model_path):
    from model import MarioModel
    model = MarioModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def get_action(model, obs, top1=True):
    with torch.no_grad():
        logits = model.forward(torch.from_numpy(np.array(
            [obs])))["logits"].cpu().detach().numpy()[0]
    if top1:
        action = np.argmax(logits)
    else:
        from scipy.special import softmax
        action = np.random.choice(np.arange(len(logits)), p=softmax(logits))
    return action


if __name__ == "__main__":
    env = make_env(6, 4)
    model = make_model("models/model.pt")

    done = False
    obs = env.reset()
    while not done:
        action = get_action(model, obs)
        obs, done, info = env.step(action)
        env.render()
        time.sleep(0.02)
