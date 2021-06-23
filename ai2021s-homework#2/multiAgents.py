# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent2(Agent):
    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]



class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """autograder
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        food_dist_list = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        if food_dist_list:
            score += 3.0 / (sum(food_dist_list) / len(food_dist_list))
            score += 10.0 / min(food_dist_list)
            score += 1.0 / max(food_dist_list)

        capsule_dist_list = [util.manhattanDistance(newPos, capsulePos) for capsulePos in successorGameState.getCapsules()]
        if capsule_dist_list:
            score += 3.0 / (sum(capsule_dist_list) / len(capsule_dist_list))
            score += 10.0 / min(capsule_dist_list)
            score += 1.0 / max(capsule_dist_list)

        for ghost in newGhostStates:
            dist = util.manhattanDistance(newPos, ghost.getPosition())
            if newScaredTimes[0] > 5:
                score += 200.0 / dist if dist else 0
            elif dist < 2:
                score += float('-Inf')

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, depth=-1, agentIndex=self.index)

    def minimax(self, gameState, depth, agentIndex):

        agentIndex = agentIndex % gameState.getNumAgents()
        depth += (agentIndex == self.index)

        if depth >= self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif agentIndex == self.index:
            value, action = self.max(gameState, depth, agentIndex)
            return action if depth == 0 else value
        else:
            return self.min(gameState, depth, agentIndex)

    def min(self, gameState, depth, agentIndex):
        min_value = float('Inf')
        for action in gameState.getLegalActions(agentIndex):
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            min_value = min(min_value, self.minimax(successorGameState, depth, agentIndex+1))
        return min_value

    def max(self, gameState, depth, agentIndex):
        max_value = float("-Inf")
        max_action = None
        for action in gameState.getLegalActions(agentIndex):
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            value = self.minimax(successorGameState, depth, agentIndex+1)
            if value > max_value:
                max_action = action
                max_value = value
        return max_value, max_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphabeta(gameState, depth=-1, agentIndex=self.index, alpha=float('-Inf'), beta=float('Inf'))

    def alphabeta(self, gameState, depth, agentIndex, alpha, beta):

        agentIndex = agentIndex % gameState.getNumAgents()
        depth += (agentIndex == self.index)

        if depth >= self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif agentIndex == self.index:
            value, action = self.max(gameState, depth, agentIndex, alpha, beta)
            return action if depth == 0 else value
        else:
            return self.min(gameState, depth, agentIndex, alpha, beta)

    def min(self, gameState, depth, agentIndex, alpha, beta):
        min_value = float('Inf')
        for action in gameState.getLegalActions(agentIndex):
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            min_value = min(min_value, self.alphabeta(successorGameState, depth, agentIndex+1, alpha, beta))
            if min_value < alpha:
                break
            beta = min(beta, min_value)
        return min_value

    def max(self, gameState, depth, agentIndex, alpha, beta):
        max_value = float("-Inf")
        max_action = None
        for action in gameState.getLegalActions(agentIndex):
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            value = self.alphabeta(successorGameState, depth, agentIndex+1, alpha, beta)
            if value > max_value:
                max_action = action
                max_value = value
            if max_value > beta:
                break
            alpha = max(alpha, max_value)
        return max_value, max_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState, depth=-1, agentIndex=self.index)

    def expectimax(self, gameState, depth, agentIndex):

        agentIndex = agentIndex % gameState.getNumAgents()
        depth += (agentIndex == self.index)

        if depth >= self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif agentIndex == self.index:
            value, action = self.max(gameState, depth, agentIndex)
            return action if depth == 0 else value
        else:
            return self.mean(gameState, depth, agentIndex)

    def mean(self, gameState, depth, agentIndex):
        mean_value = 0.0
        for action in gameState.getLegalActions(agentIndex):
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            mean_value += self.expectimax(successorGameState, depth, agentIndex+1)
        return mean_value / len(gameState.getLegalActions(agentIndex))

    def max(self, gameState, depth, agentIndex):
        max_value = float("-Inf")
        max_action = None
        for action in gameState.getLegalActions(agentIndex):
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            value = self.expectimax(successorGameState, depth, agentIndex+1)
            if value > max_value:
                max_action = action
                max_value = value
        return max_value, max_action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
