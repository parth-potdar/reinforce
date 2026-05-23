from reinforce.policy_network import PolicyNetwork

class REINFORCEAgent:
    def __init__(self, env):
        """Initialise REINFORCE Agent"""

        self.policy_network = PolicyNetwork()
    
    def select_action(self):
        """Select action from current policy based on current state"""

    def update(self):
        """Update policy parameters using REINFORCE algorithm"""
        