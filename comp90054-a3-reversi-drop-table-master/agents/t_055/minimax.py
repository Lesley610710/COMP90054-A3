import json
from json.encoder import INFINITY
from Reversi.reversi_utils import Cell
from template import Agent
from Reversi.reversi_model import ReversiGameRule
import random, time

CORNERS = [(0,0),(0,7),(7,0),(7,7)]
CORNERS_P = [(1,1),(1,6),(6,1),(6,6)]

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game_rules = ReversiGameRule(2)
    
    def PerformAction(self, state, action, agent_id):
        next_state = self.game_rules.generateSuccessor(state, action, agent_id)
        return next_state

    def GameEnd(self, state):
        if self.game_rules.getLegalActions(state,0) == ["Pass"] \
             and self.game_rules.getLegalActions(state,1) == ["Pass"]:
             return True
        else: return False

    
    def CountAll(self, state):
        score_a = self.game_rules.calScore(state, self.id)
        score_b = self.game_rules.calScore(state, 1 -self.id)
        return score_a + score_b

    def SelectAction(self,actions,game_state):
        self.game_rules.agent_colors = game_state.agent_colors
        alpha = -INFINITY
        beta = INFINITY

        countall = self.CountAll(game_state)

        #_, move = self.minimax(game_state, 2, self.id, alpha, beta)
        
        if(countall < 51):
            _, move = self.minimax(game_state, 2, self.id, alpha, beta)
        else:
            _, move = self.minimax(game_state, 4, self.id, alpha, beta)

        return move

        
    def minimax(self, state, depth, agent_id, alpha, beta):
        if self.GameEnd(state) or depth == 0:
            return self.heuristic(state), None
        best_move = None

        #max
        if agent_id == self.id:
            best_score = -INFINITY
            next_actions = self.game_rules.getLegalActions(state, agent_id)
            con_actions = list(set(next_actions).intersection(set(CORNERS)))
            if(len(con_actions) > 0):
                next_actions = con_actions
            con_actions = list(set(next_actions).difference(set(CORNERS_P)))
            if(len(con_actions) > 0):
                next_actions = con_actions

            for action in next_actions:
                next_state = self.PerformAction(state, action, agent_id)
                score, _ = self.minimax(next_state, depth - 1, 1 - agent_id, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_move = action
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score,best_move

        #min
        else:
            best_score = INFINITY
            next_actions = self.game_rules.getLegalActions(state, 1 - agent_id)
            con_actions = list(set(next_actions).intersection(set(CORNERS)))
            if(len(con_actions) > 0):
                next_actions = con_actions
            con_actions = list(set(next_actions).difference(set(CORNERS_P)))
            if(len(con_actions) > 0):
                next_actions = con_actions

            for action in next_actions:
                next_state = self.PerformAction(state, action, 1 - agent_id)
                score, _ = self.minimax(next_state, depth - 1, agent_id, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_move = action
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score,_

    def heuristic(self, state):
        age_color = self.game_rules.agent_colors[self.id]
        oppo_color = self.game_rules.agent_colors[1 - self.id]
        features = []
        #next_state = self.PerformAction(state, action, self.id)
        #agent's pieces count
        agent_score = self.game_rules.calScore(state, self.id)
        h1 = agent_score/64
        features.append(h1)
        #rival's pieces count
        rival_score = self.game_rules.calScore(state, 1 - self.id)
        h2 = rival_score/64
        features.append(h2)
        #
        h3 = 0
        h4 = 0
        h7 = 0
        h8 = 0

        board = state.board
        #pieces around corner
        age_pieces = 0
        oppo_pieces = 0
        if board[0][0] == Cell.EMPTY:
            if board[0][1] == age_color:
                age_pieces += 1
            elif board[0][1] == oppo_color:
                oppo_pieces += 1
            if board[1][1] == age_color:
                age_pieces += 1
            elif board[1][1] == oppo_color:
                oppo_pieces += 1
            if board[1][0] == age_color:
                age_pieces += 1
            elif board[1][0] == oppo_color:
                oppo_pieces += 1

        if board[0][7] == Cell.EMPTY:
            if board[0][6] == age_color:
                age_pieces += 1
            elif board[0][6] == oppo_color:
                oppo_pieces += 1
            if board[1][6] == age_color:
                age_pieces += 1
            elif board[1][6] == oppo_color:
                oppo_pieces += 1
            if board[1][7] == age_color:
                age_pieces += 1
            elif board[1][7] == oppo_color:
                oppo_pieces += 1

        if board[7][0] == Cell.EMPTY:
            if board[7][1] == age_color:
                age_pieces += 1
            elif board[7][1] == oppo_color:
                oppo_pieces += 1
            if board[6][1] == age_color:
                age_pieces += 1
            elif board[6][1] == oppo_color:
                oppo_pieces += 1
            if board[6][0] == age_color:
                age_pieces += 1
            elif board[6][0] == oppo_color:
                oppo_pieces += 1

        if board[7][7] == Cell.EMPTY:
            if board[6][7] == age_color:
                age_pieces += 1
            elif board[6][7] == oppo_color:
                oppo_pieces += 1
            if board[6][6] == age_color:
                age_pieces += 1
            elif board[6][6] == oppo_color:
                oppo_pieces += 1
            if board[7][6] == age_color:
                age_pieces += 1
            elif board[7][6] == oppo_color:
                oppo_pieces += 1

        h5 = age_pieces
        h6 = oppo_pieces

        h11 = 0
        cor1 = [0,0,7,7]
        cor2 = [0,7,7,0]
        dir1 = [0,1,0,-1]
        dir2 = [1,0,-1,0]
        stop = [0,0,0,0]
        for i in range(4):
            if state.board[CORNERS[i][0]][CORNERS[i][1]] == age_color:
                h3 += 1
                stop[i] = 1
                for j in range(1,7):
                    if board[cor1[i]+dir1[i]*j][cor2[i]+dir2[i]*j] != age_color:
                        break
                    else:
                        stop[i] = j + 1
                        h11 += 1
            if state.board[CORNERS[i][0]][CORNERS[i][1]] == Cell.EMPTY and  state.board[CORNERS_P[i][0]][CORNERS_P[i][1]] == age_color:
                h4 += 1
            if state.board[CORNERS[i][0]][CORNERS[i][1]] == oppo_color:
                h7 += 1
            if state.board[CORNERS[i][0]][CORNERS[i][1]] == Cell.EMPTY and  state.board[CORNERS_P[i][0]][CORNERS_P[i][1]] == oppo_color:
                h8 += 1
        for i in range(4):
            if board[cor1[i]][cor2[i]] == age_color:
                for j in range(1,7-stop[i-1]):
                    if board[cor1[i]-dir1[i-1]*j][cor2[i]-dir2[i-1]*j] != age_color:
                        break
                    else:
                        h11 += 1
        features.append(h3/4)
        features.append(h4/4)
        #Edge
        features.append(h5/4)
        features.append(h6/4)
        #Rival Corner
        features.append(h7/4)
        features.append(h8/4)
        #mobility
        h9 = len(self.game_rules.getLegalActions(state, self.id))
        h10 = len(self.game_rules.getLegalActions(state, 1 - self.id))
        features.append(h9/4)
        features.append(h10/4)
        #print(features)
        return 0.02 * h1 + (-0.02) * h2 + 1.2 * h3 + (-0.9) * h4 + (-1.5) * h5 + 0.9 * h6 + (-1 * h7) + 0.65 * h8 + 0.25 * h9 + (-0.4 * h10)  + 0.5 * h11