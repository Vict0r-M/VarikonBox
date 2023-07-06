import torch
import torch.nn.functional as F
import numpy as np
import random

class Varikon:
    def __init__(self):
        # Choose random position for the empty cubicle:
        self.empty_pos = torch.tensor([random.choice([0, 2, 5, 7, 12, 14, 17, 19])])
        self.empty_cubicle = self.empty_pos.clone() # Initialize empty cubicle
        self.orientations = self.get_orientations() # Get valid orientations
        self.actionsList = ["U", "F", "L", "D", "B", "R"]

    def get_orientations(self):
        # Define orientations based on their possible positions in the solved state:
        orientations = torch.tensor([[0, 1, 3, 8], # Cubicle with FRU squares blue
                                     [1, 2, 4, 9], # Cubicle with FLU squares blue
                                     [3, 5, 6, 10], # Cubicle with BRU squares blue
                                     [4, 6, 7, 11], # Cubicle with BLU squares blue
                                     [8, 12, 13, 15], # Cubicle with FRD squares blue
                                     [9, 13, 14, 16], # Cubicle with FLD squares blue
                                     [10, 15, 17, 18], # Cubicle with BRD squares blue
                                     [11, 16, 18, 19]], # Cubicle with BLD squares blue
                                     dtype=torch.long) # Define as long type tensor

        empty_pos_mapping = { # Define mapping empty_pos -> unusable orientation
            0: 0,  # FRU
            2: 1,  # FLU
            5: 2,  # BRU
            7: 3,  # BLU
            12: 4,  # FRD
            14: 5,  # FLD
            17: 6,  # BRD
            19: 7  # BLD
        }

        # Find and remove unusable orientation
        invalid_cubicle_index = empty_pos_mapping[self.empty_pos.item()]
        orientations = torch.cat([orientations[:invalid_cubicle_index],
                                  orientations[invalid_cubicle_index+1:]])

        return orientations

    def getSolvedState(self):
        state = torch.zeros(self.N ** 2 * 6, dtype=torch.uint8)
        for i in range(6):
            state[self.N ** 2 * i: self.N ** 2 * (i + 1)] = i

        return state

    def checkIfSolved(self, states):
        return torch.all(torch.stack([i in states[j] for j in range(20) for i in states[j]]).reshape(states.shape[0], -1), axis=1)

    def checkIfSolvedSingle(self, state):
        return torch.all(torch.tensor([i in state for i in range(20)]))


    def doAction(self, action, state=None):
        assert action in self.actionsList
    
        if state is None:
            state = self.state
    
        # Define the actions
        actions = {
            0: {"R": [(1, 0), (2, 1), (self.empty_cubicle, 2)], "F": [(3, 0), (5, 3), (self.empty_cubicle, 5)], "U": [(8, 0), (12, 8), (self.empty_cubicle, 12)]},
            2: {"L": [(1, 2), (0, 1), (self.empty_cubicle, 0)], "F": [(4, 2), (7, 4), (self.empty_cubicle, 7)], "U": [(9, 2), (14, 9), (self.empty_cubicle, 14)]},
            5: {"B": [(3, 5), (0, 3), (self.empty_cubicle, 0)], "R": [(6, 5), (7, 6), (self.empty_cubicle, 7)], "U": [(10, 5), (17, 10), (self.empty_cubicle, 17)]},
            7: {"L": [(6, 7), (5, 6), (self.empty_cubicle, 5)], "B": [(4, 7), (2, 4), (self.empty_cubicle, 2)], "U": [(11, 7), (19, 11), (self.empty_cubicle, 19)]},
            12: {"R": [(13, 12), (14, 13), (self.empty_cubicle, 14)], "F": [(15, 12), (17, 15), (self.empty_cubicle, 17)], "D": [(8, 12), (0, 8), (self.empty_cubicle, 0)]},
            14: {"L": [(13, 14), (12, 13), (self.empty_cubicle, 12)], "F": [(16, 14), (19, 16), (self.empty_cubicle, 19)], "D": [(9, 14), (2, 9), (self.empty_cubicle, 2)]},
            17: {"R": [(18, 17), (19, 18), (self.empty_cubicle, 19)], "B": [(15, 17), (12, 15), (self.empty_cubicle, 12)], "D": [(10, 17), (5, 10), (self.empty_cubicle, 5)]},
            19: {"L": [(18, 19), (17, 18), (self.empty_cubicle, 17)], "B": [(16, 19), (14, 16), (self.empty_cubicle, 14)], "D": [(11, 19), (7, 11), (self.empty_cubicle, 7)]}
        }
    
        # Get the current position of the empty cubicle
        empty_pos = (state == self.empty_cubicle).nonzero(as_tuple=True)[0][0]
    
        # Get the actions for the current position of the empty cubicle
        current_actions = actions[empty_pos]
    
        # Check if the action is valid for the current position of the empty cubicle
        if action not in current_actions:
            raise ValueError(f"Invalid action {action} for current position {empty_pos}")
    
        # Perform the action
        for old_pos, new_pos in current_actions[action]:
            state[new_pos] = state[old_pos]
    
        return state

    def nextState(self, states, actions,):
        return states.gather(1, self.nextStateMat.index_select(0, torch.as_tensor(actions)),)

    def generateScramble(self):
        state = torch.empty((20, 4), dtype=torch.long) # Initialize empty state
        # Allocate empty cubicle to a random position:
        state[torch.randint(0, 20, (1,))] = self.empty_cubicle
        used_indexes = self.empty_cubicle.clone() # Initialize tensor of used indexes

        while (state == 0).any(): # While there are still positions to fill
            # Choose random valid cubicle
            cubicle = self.orientations[torch.randint(0, self.orientations.shape[0], (1,))]

            for pos_index in cubicle[0]: # Check if position index was used
                if (used_indexes == pos_index).sum() == 0:
                    # Add cubicle to state:
                    state[(state == 0).nonzero(as_tuple=True)[0][0]] = cubicle
                    # Add position index to used indexes:
                    used_indexes = torch.cat([used_indexes, cubicle[0]], dim=0)
                    break # Break for variety

        return state # Return a single scrambled state

    def generateScrambles(self, num_states): # Generate num_states states
        states = torch.stack([self.generateScramble() for _ in range(num_states)])
        return states # Return num_states scrambled states

    def exploreNextStates(self, states):

        validStates = torch.tensor(
            [True] * self.num_actions).repeat(states.shape[0], 1)

        nextStates = states.repeat_interleave(self.num_actions, dim=0).gather(
            1,
            self.nextStateMat.index_select(
                0,
                torch.as_tensor(
                    np.tile(
                        np.arange(0, self.num_actions,
                                  dtype=np.int64), states.shape[0],
                    ),
                ),
            ),
        )

        nextStates = nextStates.view(states.shape[0], self.num_actions, -1)

        goals = self.checkIfSolved(nextStates.view(-1, self.N ** 2 * 6)).view(-1, self.num_actions)

        return nextStates, validStates, goals

    def NextStateSpotToAction(self, i):
        return self.actions_list[i]

    @staticmethod
    def oneHotEncoding(states):
        return F.one_hot(states.view(-1).long(), 6).view(-1, states.shape[1]*6)