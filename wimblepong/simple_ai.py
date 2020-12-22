from wimblepong import Wimblepong
import random


class SimpleAi(object):
    def __init__(self, env, player_id=1):
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")
        self.env = env
        # Set the player id that determines on which side the ai is going to play
        self.player_id = player_id  
        # Ball prediction error, introduce noise such that SimpleAI reflects not
        # only in straight lines
        self.bpe = 4
        self.name = "SimpleAI"
        self.personality = random.choice([1, 2, 3, 1, 4, 1, 5, 6, 1, 1, 3])
        print('I am',self.personality)
        self.personality=1
        self.timestep = 0
        self.counter = 0
        self.name = "SimpleAI_%s" % (self.personality)
        self.target = 0
        self.anterior = 0
    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def get_action(self, ob=None):
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        """
        # Get the player id from the environment
        player = self.env.player1 if self.player_id == 1 else self.env.player2

        # Get own position in the game arena
        my_y = player.y
        my_x = player.x
        # Get the ball position in the game arena
        ball_y = self.env.ball.y + (random.random()*self.bpe-self.bpe/2)
        ball_x = self.env.ball.x
        vel_x = self.env.ball.vector[0]
        if self.counter%30000:
            self.bpe=self.bpe-1
        self.counter += 1
        if self.bpe < 3:
            self.bpe=2
        self.bpe=4
        #if self.target < 2:
        #    self.target=0
        """                 randomnumber=0
                self.timestep += 1

                randomnumber= int(self.target*random.random())
                self.counter += 1 """
        # Compute the difference in position and try to minimize it
        if self.personality==1:
            y_diff = my_y - ball_y

            if abs(y_diff) < 2:
                action = 0  # Stay
            else:
                if y_diff > 0:
                    action = self.env.MOVE_UP  # Up
                else:
                    action = self.env.MOVE_DOWN  # Down
        if self.personality==2:
            y_diff = my_y - ball_y
            x_diff = my_x - ball_x
            #print(x_diff)

            if abs(y_diff) < 2:
                action = 0  # Stay
            else:
                if y_diff > 0:
                    action = self.env.MOVE_UP  # Up
                else:
                    action = self.env.MOVE_DOWN  # Down
            if abs(x_diff) > 70:
                #print('here')
                action = random.choice([1, 2, 0 ,0, 0])


        if self.personality==3:
            y_diff = my_y - (ball_y+random.randrange(-10,10))


            if abs(y_diff) < 2:
                action = 0  # Stay
            else:
                if y_diff > 0:
                    action = self.env.MOVE_UP  # Up
                else:
                    action = self.env.MOVE_DOWN  # Down
            if (vel_x<0):
                action = random.choice([1, 2, 0 ,0, 0, 0])
        if self.personality==4:
            y_diff = my_y - ball_y
            x_diff_before = my_x - self.anterior
            x_diff = my_x - ball_x


            if abs(y_diff) < 2:
                action = 0  # Stay
            else:
                if y_diff > 0:
                    action = self.env.MOVE_UP  # Up
                else:
                    action = self.env.MOVE_DOWN  # Down
            if abs(x_diff)>abs(x_diff_before) :
                action = random.choice([1, 2, 0 ,0, 0, 0, 0, 0])
            self.anterior = ball_x
        if self.personality==5:
            y_diff = my_y - ball_y+10
            x_diff_before = my_x - self.anterior
            x_diff = my_x - ball_x


            if abs(y_diff) < 2:
                action = 0  # Stay
            else:
                if y_diff > 0:
                    action = self.env.MOVE_UP  # Up
                else:
                    action = self.env.MOVE_DOWN  # Down
            if abs(x_diff)>abs(x_diff_before) :
                action = random.choice([1, 2, 0 ,0, 0, 0, 0, 0])
            self.anterior = ball_x
        if self.personality==6:
            y_diff = my_y - ball_y-10
            x_diff_before = my_x - self.anterior
            x_diff = my_x - ball_x


            if abs(y_diff) < 2:
                action = 0  # Stay
            else:
                if y_diff > 0:
                    action = self.env.MOVE_UP  # Up
                else:
                    action = self.env.MOVE_DOWN  # Down
            if abs(x_diff)>abs(x_diff_before) :
                action = random.choice([1, 2, 0 ,0, 0, 0, 0, 0])
            self.anterior = ball_x
        return action

    def reset(self):
        # Nothing to done for now...
        return


