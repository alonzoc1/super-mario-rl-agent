
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
# from gym.wrappers.monitoring import video_recorder
import argparse
import time
import numpy as np
from nes_py.wrappers import JoypadSpace
from tqdm import tqdm
from gymnasium.wrappers.monitoring import video_recorder
import frame_skipper
import agent_dqn

TEST_EPISODES = 10 # How many episodes to go through when testing
MAX_ITER = 11000 # Max iterations when training
MODEL_PATH = './models/'
VIDEO_PATH = './test_videos/'
GRAPH_PATH = './graphs/'

def parse():
    parser = argparse.ArgumentParser(description="Super Mario RL Project 4")
    parser.add_argument('--train', action='store_true', help='whether train DQN')
    parser.add_argument('--test', action='store_true', help='whether test DQN')
    parser.add_argument('--model', default=None, help='model file name')
    parser.add_argument('--display', action='store_true', help='whether to show the gameplay, do not use with record video')
    parser.add_argument('--record_video', action='store_true', help='whether to record video during testing, do not use with display')
    args = parser.parse_args()
    return args


def setup_env(args):
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='human' if args.display else 'rgb_array', apply_api_compatibility=True)
    # all of these help with performance
    env = JoypadSpace(env, RIGHT_ONLY) # agent only needs to go right
    env = frame_skipper.FrameSkipper(env, skip=4)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    return env

def test(args, env, video_path, agent, start_time):
    rewards = []
    vid = None
    if args.record_video:
        vid = video_recorder.VideoRecorder(env=env.env, path=video_path)
    
    for _ in tqdm(range(TEST_EPISODES)):
        episode_reward = 0.0
        truncated = False
        state, _ = env.reset()
        terminated = False
        while not terminated and not truncated:
            action = agent.take_action(state, test=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if args.record_video:
                vid.capture_frame()

        if truncated:
            break

    rewards.append(episode_reward)

    if args.record_video:
        vid.close()  # Ensure the video recorder is properly closed

    env.close()

    print('Run %d episodes' % (TEST_EPISODES))
    print('Mean:', np.mean(rewards))
    print('rewards', rewards)
    print('running time', time.time()-start_time)

def run(args, env):
    time_now = time.time()
    time_now_str = str(time_now)
    if (args.test):
        if (args.model == None):
            print("Please provide the model file name to test")
            return
        # Run test with provided model
        model = MODEL_PATH + str(args.model)
        agent = agent_dqn.AgentDQN(env, args, model, time_now)
        vid_path = VIDEO_PATH + 'video_' + time_now_str + '.mp4'
        test(args, env, vid_path, agent, time_now)
    elif (args.train):
        continue_training = False
        if (args.model):
            continue_training = True
        model = MODEL_PATH + ('model_' + time_now_str + '.pth' if args.model is None else args.model)
        agent = agent_dqn.AgentDQN(env, args, model, time_now, continue_training)
        agent.train(MAX_ITER)
    else:
        print("No action provided, exiting...")
        return

if __name__ == "__main__":
    args = parse()
    env = setup_env(args)
    run(args, env)
