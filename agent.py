import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from racecar.qnetwork import Qnetwork


class Agent(object):
    def __init__(self, gamma, epsilon, epsilonDecay, epsilonMin, alpha, maxMemSize, actionSpace, targetReplaceCount, episodeEnd):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin
        self.alpha = alpha
        self.maxMemSize = maxMemSize
        self.actionSpace = actionSpace
        self.episodeEnd = episodeEnd
        self.targetReplaceCount = targetReplaceCount

        self.steps = 0
        self.stepCounter = 0
        self.memory = []
        self.memCounter = 0

        self.Qevaluation = Qnetwork(len(self.actionSpace), alpha)
        self.Qprediction = Qnetwork(len(self.actionSpace), alpha)

    def storeTransition(self, state, action, reward, state_):
        if self.memCounter < self.maxMemSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCounter % self.maxMemSize] = [state, action, reward, state_]
        self.memCounter += 1

    def chooseAction(self, observation):
        randomActionChance = np.random.random()

        if randomActionChance < self.epsilon:
            action = np.random.choice(self.actionSpace)

        else:
            action = torch.argmax(self.Qevaluation.forward(observation)[1].item())

        self.steps += 1
        return action

    def learn(self, batchSize):
        self.Qevaluation.optimizer.zero_grad()
        if self.targetReplaceCount is not None and self.stepCounter%self.targetReplaceCount == 0:
            #TODO remove ? not used ?
            self.Qprediction.load_state_dict(self.Qevaluation.state_dict())

        minibatch = []
        if self.memCounter < batchSize:
            minibatch = self.memory[0:self.memCounter-1]
        else:
            batchStart = np.random.randint(0, self.memCounter - batchSize)
            minibatch = self.memory[batchStart: batchStart + batchSize]

        memory = np.array(minibatch)

        evaluation = self.Qevaluation.forward(list(memory[:, 0][:])).to(self.Qevaluation.device)
        prediction = self.Qprediction.forward(list(memory[:, 3][:])).to(self.Qprediction.device)

        maxAction = torch.argmax(prediction, dim=1).to(self.Qevaluation.device)
        rewards = torch.Tensor(list(memory[:, 2])).to(self.Qevaluation.device)
        target = evaluation

        # BELLMAN
        targetCalc = rewards + self.gamma * torch.max(prediction[1])

        target[:, maxAction] = rewards + self.gamma * torch.max(prediction[1])

        #LOSS Function
        loss = self.Qevaluation.loss(target, prediction).to(self.Qevaluation.device)
        loss.backward()

        self.Qevaluation.optimizer.step()
        self.stepCounter += 1
        # self.Qevaluation.optimizer.zero_grad()

        # EPSILON DECAY
        if self.steps > 500:
            if self.epsilon > self.epsilonMin:
                self.epsilon *= self.epsilonDecay
            else:
                self.epsilon = self.epsilonMin



