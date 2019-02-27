import gym
from racecar.qnetwork import Qnetwork
from racecar.agent import Agent
import numpy as np
import time
from racecar.utils import saveImage, showImage, standardPreprocess

np.set_printoptions(threshold=np.nan)


env = gym.make('CarRacing-v0')
env.reset()
discreteActions = np.array([[0.0, 1.0, 0.0], [1.0, 0.3, 0], [-1.0, 0.3, 0.0], [0.0, 0.0, 0.8]])
actionSpace = [0, 1, 2 , 3]

# act = disc_actions[0]

agent = Agent(  gamma=0.95,
                epsilon = 1.0,
                epsilonDecay=0.99,
                epsilonMin=0.1,
                alpha = 0.003,
                maxMemSize=150,
                targetReplaceCount=None, #don't use replacement, check this out later
                actionSpace=actionSpace,
                episodeEnd=0.05
              )
# INITIALIZE AGENT MEMORY TODO: is this still necessary & isn't this already handled in agent.storeTransition?
while agent.memCounter < agent.maxMemSize:
    print(agent.memCounter)
    observation = env.reset()
    done = False
    initFrameCounter = 0
    while not done:
        actionIndex = np.random.randint(0, 3)
        action = discreteActions[actionIndex]
        observation_, reward, done, info = env.step(action)
        agent.storeTransition(np.mean(observation, axis=2), action, reward, np.mean(observation, axis=2))

        observation = observation_

        initFrameCounter+= 1
        if initFrameCounter > 50:
            done = True
print("memory initialized")

scores = []
epsilonHistory = []
numberOfGames = 10
batchSize = 32

for i in range(numberOfGames):
    print("starting game ", str(i), " w/ epsilon ", agent.epsilon )
    epsilonHistory.append(agent.epsilon)
    done = False
    observation = env.reset()

    # step past the zooming in the beginning of the game
    for z in range(60):
        env.step([0,0,0])
        env.render()


    # frames = [observation]
    frames = []
    score = 0
    lastAction = 0
    frameCounter = 0

    while not done:
        if len(frames) == 3:
            action = agent.chooseAction(frames)
            frames = []
        else:
            action = lastAction

        # Use the chosen action as an index to choose the real action
        act = discreteActions[action]
        observation_, reward, done, info = env.step(act)
        score += reward

        frames.append(np.mean(observation_, axis=2)) #observation_[0:80, 0:96]
        #TODO too long driving offroad
        #for now now maximum render 50 frames
        if(frameCounter > 150):
            done = True
            frameCounter = 0
        frameCounter += 1

        agent.storeTransition(np.mean(observation, axis=2), action, reward, observation)
        observation = observation_
        agent.learn(batchSize)
        lastAction = action
        env.render()

# for i in range(0, 50 ):
#     observation_, reward, done, info = env.step([0.0, 1.0, 0.0])
#     env.render()
#
# saveImage(observation_[0:80, 0:95])


# image = np.array(standardPreprocess(observation_)).astype(np.uint8)

#TODO TRY TO PRINT GREYSCALE ARRAY (FOR SHOW)

# grayImage = []
# for m in range(len(image)):
#     grayImage.append()
#     for n in range(len(image[m])):
#         grayImage[m][n] = 5
#
# print(grayImage[0,0])





# agent = Agent

'''TESTING NETWORK'''
# for i in range(0, 15):
#     observation_, reward, done, info = env.step([0.0, 1.0, 0.0])
#     env.render()
#
# qnet = Qnetwork(len(discreteActions), 0.01)
#
#
# print(observation_)
# qnet.forward(observation_[0:80, 0:96])

''' some things'''
#  96 x 96
# print("input image is: " + str(len(observation_)) + " x " + str(len(observation_[0])))

# for _ in range(1000):
#     time.sleep(10)
#     observation_, reward, done, info = env.step(act)
#     print

    #env.step(env.action_space.sample()) # take a random action
