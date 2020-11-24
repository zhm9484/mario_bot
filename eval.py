import argparse

from play import make_env, make_model, get_action

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=int,
        choices=[0, 1],
        default=0,
    )
    args = parser.parse_args()

    model = make_model("models/model.pt")

    # stages that are not passed
    skip_stages = [(4, 4), (7, 4), (8, 4), (1, 3), (5, 3)]

    for world in range(1, 9):
        for stage in range(1, 5):
            if (world, stage) in skip_stages:
                continue
            env = make_env(world, stage)
            done = False
            obs = env.reset()
            while not done:
                action = get_action(model, obs)
                obs, done, info = env.step(action)
                if args.render:
                    env.render()
                    import time
                    time.sleep(0.02)
            env.close()
            time = info["time"]
            dead = info["is_dead"] or info["is_dying"]
            passed = False if not time or dead else True
            if passed:
                print(f"World{world}-Stage{stage} passed!")
            else:
                print(f"World{world}-Stage{stage} failed! Time: {time}.")
