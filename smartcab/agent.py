import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    # The init function now takes alpha and gamma as parameters
    def __init__(self, env,alpha, gamma):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
	self.possible_action = [None, 'forward', 'left', 'right']
        # TODO: Initialize any additional variables here
        # Q values 
	self.qtable ={}
	# Learning rate 
	self.alpha = alpha/100.0
	# Futur reward valuation
	self.gamma = gamma/100.0
	# List of successful trial
	self.successes = []
	# Counter
	self.i=1
	# Sum of all rewards obtained through trial
	self.sum_reward=0
	
		
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # Assume failure (0) of trial
        self.successes.append(0)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        # Take input & waypoints as definition of the state
        self.state = (inputs,self.next_waypoint)#,deadline)
        # TODO: Select action according to your policy
        # Call argmax function	
        action = self.argmax(str(self.state))
        # Execute action and get reward
        reward = self.env.act(self, action)
        # Increment sum of reward
        self.sum_reward += reward

        # TODO: Learn policy based on state, action, reward
        # Calculate and save the new qval (step by step for ease of understanding)
        val = self.qtable.get(str(self.state)+str(action),0)
        val = val - self.alpha*val
        val = val + self.alpha*reward
        new_state = (self.env.sense(self),self.planner.next_waypoint())#,self.env.get_deadline(self))
        next_a = self.argmax(new_state)
        val = val + self.alpha*self.gamma*self.qtable.get(str(new_state)+str(next_a),0.0)
        self.qtable[str(self.state)+str(action)]=val


        # Check sucess or failure
        location = self.env.agent_states[self]["location"] 
        destination = self.env.agent_states[self]["destination"]
        if location==destination:
            self.successes[-1]=1


        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    # argmax function return best reward action
    def argmax(self,current_state):
        # initialize best action to a random value
        best_action = random.choice(self.possible_action)
        key=str(current_state)+str(best_action)
        # query Q table for the value, if doesnt exists return 0
        qval = self.qtable.get(key,0)
        # Iterate through all possible action to determine the best one
        for action in self.possible_action:
            if self.qtable.get(str(current_state)+str(action),0) > qval:
                best_action = action
                qval = self.qtable.get(str(current_state)+str(best_action),0)
            else:
                continue
        return best_action

def run():
    """Run the agent for a finite number of trials."""
    success=[]
    # Iterate through alpha and gamma to find the best combination
    for alpha in range(0,100,5):
        for gamma in range(0,100,5):
            # test current alhpa/gamma combination
            a=run_sim(alpha,gamma,0.005,False)
            success_rate = a.successes.count(1)
            sum_reward= a.sum_reward
            # Add data to the success list with alpha gamma, success rate (%) and sum of reward of the current combination
            success.append([alpha, gamma, success_rate, sum_reward])
            a=None
            e=None
    print "Results of alpha/gamma research", success
    # Extract best success rate from results
    max_rate = max(success, key=lambda x: x[2])[2]
    # Create list of all combination that holds the best success rate (may be > 1)
    lim_list = [item for item in success if item[2]==max_rate]
    # Get the best combination by taking the max of sum_rewards among the best success_rate combination
    best_settings = max(lim_list, key=lambda x: x[3])
    # Extract alpha from best_settings
    best_alpha = best_settings[0]
    # Extract gamma from best_settings
    best_gamma = best_settings[1]
    print "Best rate found: ", max_rate
    print "Potential combination: ", lim_list
    print "Best alpha: ", alpha
    print "Best gamma: ", gamma
    print "Running graphical simulation with best settings"
    best = run_sim(best_alpha,best_gamma,0.5,True)
    print "Success rate from graphical simulation: ",best.successes.count(1)
    
def run_sim(alpha,gamma,time_step,disp):
            # Set up environment and agent
            # create environment (also adds some dummy traffic)
            e = Environment()
            a = e.create_agent(LearningAgent,alpha,gamma)  # create agent
            e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
            # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

            # Now simulate it
            sim = Simulator(e, update_delay=time_step, display=disp)  # create simulator (uses pygame when display=True, if available)
            # NOTE: To speed up simulation, reduce update_delay and/or set display=False

            sim.run(n_trials=100)  # run for a specified number of trials
            return a

if __name__ == '__main__':
    run()
