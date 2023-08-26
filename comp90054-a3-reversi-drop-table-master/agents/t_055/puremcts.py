import collections
from json.encoder import INFINITY
import math
from template import Agent
from Reversi.reversi_model import ReversiGameRule
import random, time
import numpy as np

Gamma = 0.8
Epsilon = 0.9
CORNERS = [(0,0),(0,7),(7,0),(7,7)]
CORNERS_P = [(1,1),(1,6),(6,1),(6,6)]

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game_rules = ReversiGameRule(2)

    def GameEnd(self, state):
        if self.game_rules.getLegalActions(state,0) == ["Pass"] \
             and self.game_rules.getLegalActions(state,1) == ["Pass"]:
             return True
        else: return False
    
    def CalScore(self, state, agent_id):
        next_score = self.game_rules.calScore(state, agent_id)
        return next_score

    def PerformActionScore(self, state, action, agent_id):
        next_state = self.game_rules.generateSuccessor(state, action, agent_id)
        next_score = self.game_rules.calScore(next_state, agent_id)
        return next_state, next_score

    
    def SelectAction(self,actions,game_state):
        self.game_rules.agent_colors = game_state.agent_colors
        # count = 0
        start_time = time.time()

        solution = random.choice(actions)

        val_ro_dict = collections.defaultdict(dict)
        value_dict = dict()
        ncount_dict = dict()
        expanded_action_s = dict()
        parent_dict = dict()

        t_root_state = 'root'
        #whether the state is fully expanded
        def is_fully_expanded(t_state, actions):
            if t_state in expanded_action_s:
                expanded_actions =  expanded_action_s[t_state]
                return list(set(actions).difference(set(expanded_actions)))
            else:
                return actions 

        def SelectBestAct(state, actions):
            stateLength = len(state)
            if(len(val_ro_dict[state]) != 0):
                maxp = max(val_ro_dict[state], key=lambda k: val_ro_dict[state][k])
                emptyTuple = tuple()
                if(maxp[stateLength] != 'P'):
                    emptyTuple = emptyTuple + (int(maxp[stateLength]),) + (int(maxp[stateLength + 1]),)
                    return emptyTuple
                else:
                    return random.choice(actions)
            else:
                    return random.choice(actions)

            
        while time.time() - start_time < 0.97:
            # count += 1
            state = game_state
            new_actions = actions
            t_cur_state = t_root_state
            t_def_state = t_cur_state
            vis_queue = Queue()
            reward = 0

            #select
            #if fully expand, select a state to expand
            while len(is_fully_expanded(t_cur_state, new_actions)) == 0 and not self.GameEnd(state):
                if time.time() - start_time > 0.97:
                    solution = SelectBestAct(t_root_state, actions)
                    return solution
                
                if(random.uniform(0,1) < 1 - Epsilon) and (t_cur_state in val_ro_dict):
                    cur_action = SelectBestAct(t_cur_state, new_actions)
                else:
                    cur_action = random.choice(new_actions)
                next_state = self.game_rules.generateSuccessor(state, cur_action, self.id)
                q = (t_cur_state, cur_action)
                vis_queue.push(q)
                #rival's turn
                rival_new_actions = self.game_rules.getLegalActions(next_state, 1 - self.id)
                rival_max_score = 0
                rival_best_state = next_state
                for rival_action in rival_new_actions:
                    rival_best_action = rival_action
                    rival_next_state, rival_next_score = self.PerformActionScore(next_state, rival_action, 1 - self.id)
                    if rival_next_score > rival_max_score:
                        rival_max_score = rival_next_score
                        rival_best_state = rival_next_state
                        rival_best_action = rival_action
                #iteration
                t_def_state = t_cur_state
                t_cur_state = t_cur_state + str(cur_action[0]) + str(cur_action[1]) + str(rival_best_action[0]) + str(rival_best_action[1])
                new_actions = self.game_rules.getLegalActions(rival_best_state, self.id)
                state = rival_best_state

            #Expand
            available_actions = is_fully_expanded(t_cur_state, new_actions)
            if(len(available_actions)) == 0:
                cur_action = random.choice(new_actions)
            else:
                cur_action = random.choice(available_actions)
            
            if t_cur_state in expanded_action_s:
                expanded_action_s[t_cur_state].append(cur_action)
            else:
                expanded_action_s[t_cur_state] = [cur_action]
            q = (t_cur_state, cur_action)
            vis_queue.push(q)
            next_state = self.game_rules.generateSuccessor(state, cur_action, self.id)
            #rival's turn
            rival_new_actions = self.game_rules.getLegalActions(next_state, 1 - self.id)
            rival_max_score = 0
            rival_best_state = next_state
            for rival_action in rival_new_actions:
                rival_next_state, rival_next_score = self.PerformActionScore(next_state, rival_action, 1 - self.id)
                if rival_next_score > rival_max_score:
                    rival_max_score = rival_next_score
                    rival_best_state = rival_next_state
                    rival_best_action = rival_action
            t_cur_state = t_cur_state + str(cur_action[0]) + str(cur_action[1]) + str(rival_best_action[0]) + str(rival_best_action[1])
            sim_actions = self.game_rules.getLegalActions(rival_best_state, self.id)
            state = rival_best_state

            #Simulation
            #initialize length of the simulation process
            sim_length = 0
            mobility = 0
            while not self.GameEnd(state):
                sim_length += 1
                mobility += len(self.game_rules.getLegalActions(state, self.id))
                if time.time() - start_time > 0.97:
                    solution = SelectBestAct(t_root_state, actions)
                    return solution
                cur_action = random.choice(sim_actions)
                next_state = self.game_rules.generateSuccessor(state, cur_action, self.id)
                #rival's turn
                rival_new_actions = self.game_rules.getLegalActions(next_state, 1 - self.id)
                rival_best_action = random.choice(rival_new_actions)
                rival_best_state = self.game_rules.generateSuccessor(next_state, rival_best_action, 1 - self.id)
                #iteration
                sim_actions = self.game_rules.getLegalActions(rival_best_state, self.id)
                state = rival_best_state
            # if(sim_length == 0):
            #     reward = self.CalScore(state, self.id)
            # else:
            #     reward = self.CalScore(state, self.id) + mobility/(2 * sim_length)
            self_score = self.CalScore(state, self.id)
            rival_score = self.CalScore(state, 1 - self.id)
            if self_score > rival_score:
                reward = 100 + self_score - rival_score
            else:
                reward = -(100 + rival_score - self_score)

            #Backpropagate
            cur_value = reward*(Gamma **sim_length)
            while not vis_queue.isEmpty() and time.time() - start_time < 0.97:
                t_state, cur_action = vis_queue.pop()
                if t_state in ncount_dict:
                    ncount_dict[t_state] += 1
                    #if cur_value > value_dict[t_state]:
                    update_value = value_dict[t_state] + (1/ncount_dict[t_state])*(cur_value - value_dict[t_state])
                    value_dict[t_state] = update_value
                    if len(t_state) == (len(t_def_state) + 4):
                        val_ro_dict[t_def_state][t_state] = update_value
                    elif t_state in parent_dict:
                        val_ro_dict[parent_dict[t_state]][t_state] = update_value
                else:
                    ncount_dict[t_state] = 1
                    value_dict[t_state] = cur_value
                    if len(t_state) == (len(t_def_state) + 4):
                        val_ro_dict[t_def_state][t_state] = cur_value
                        parent_dict[t_state] = t_def_state
                cur_value *= Gamma

        solution = SelectBestAct(t_root_state, actions)
        return solution

    

class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0


