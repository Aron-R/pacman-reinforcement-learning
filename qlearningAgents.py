from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math

class QLearningAgent(ReinforcementAgent):
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.qvalues = util.Counter()

    def getQValue(self, state, action):
        if (state, action) in self.qvalues.keys():
            return self.qvalues[(state, action)]
        else:
            return 0.0

    def computeValueFromQValues(self, state):
        all_actions = self.getLegalActions(state)
        if len(all_actions) == 0:
            return 0.0
        qvalues = [self.getQValue(state, action) for action in all_actions]
        return max(qvalues)

    def computeActionFromQValues(self, state):
        all_actions = self.getLegalActions(state)
        if len(all_actions) == 0:
            return None
        limit = self.getValue(state)
        possible_actions = [
            action
            for action in all_actions
            if self.getQValue(state, action) == limit
        ]
        return random.choice(possible_actions)

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        qupdate = self.getQValue(state, action) + self.alpha * (
            reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)
        )
        self.qvalues[(state, action)] = qupdate

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

class PacmanQAgent(QLearningAgent):
    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action

class ApproximateQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getWeight(self, feature):
        return self.weights[feature]

    def getQValue(self, state, action):
        all_features = self.featExtractor.getFeatures(state, action)
        return sum(
            all_features[feature] * self.getWeight(feature)
            for feature in all_features
        )

    def update(self, state, action, nextState, reward):
        all_features = self.featExtractor.getFeatures(state, action)
        difference = reward + (self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        for feature in all_features:
            self.weights[feature] = self.getWeight(feature) + (self.alpha * difference * all_features[feature])

    def final(self, state):
        PacmanQAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            print("Weights:", self.weights)
            for state in self.qvalues:
                for action in self.qvalues[state]:
                    print(f"Q({state}, {action}): {self.qvalues[state][action]}")
