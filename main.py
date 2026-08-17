from agent import Agent
from environment import SnakeGameAI
from plothelper import plot

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    # Init agent and env
    agent = Agent()
    game = SnakeGameAI()

    print("Training started! Close pygame window or press Ctrl+C to stop training.")

    while True:
        # 1: get current game state from env
        old_state = agent.get_state(game)

        # 2: get action from agent
        final_move = agent.get_action(old_state)

        # 3: perform move and get feedback
        reward, done, score = game.step(final_move)

        # 4: get new state
        new_state = agent.get_state(game)

        # 5: train short memory
        agent.train_short_memory(old_state, final_move, reward, new_state, done)

        # 6: remember experience
        agent.remember(old_state, final_move, reward, new_state, done)

        # 7: check if game over
        if done:
            # reset game layout for next iteration
            game.reset()
            agent.n_games += 1

            # train long-term memory (experience replay on batches of experiences)
            agent.train_long_memory()

            agent.total_score += score

            # Reads and checks against the record value tied to the agent instance
            if score > agent.record:
                agent.record = score
                agent.model.save()

                print(f"ALL-TIME HIGH SCORE BREAKTHROUGH! Model saved at score: {agent.record}")

            # Update the tiny text file to save the score and current game count
            with open('./model/record.txt', 'w') as f:
                f.write(f"{agent.record}\n{agent.n_games}\n{agent.total_score}")

            mean_score = agent.total_score / agent.n_games

            # Print live stats to terminal
            print(f'Game: {agent.n_games}, Score: {score}, Record: {agent.record}, Mean Score: {mean_score:.2f}')

            plot_scores.append(score)
            plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
