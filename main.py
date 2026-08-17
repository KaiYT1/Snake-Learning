from agent import Agent
from environment import SnakeGameAI
from plothelper import plot
import sys


def run_pipeline(eval_mode=False):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0

    # Init agent and env
    agent = Agent()

    if eval_mode:
        print("EVALUATION MODE: testing the best saved brain with exploration disabled.")
        game = SnakeGameAI(render_gui=True)
    else:
        print("TRAINING MODE: optimizing parameters.")
        game = SnakeGameAI(render_gui=False)

    while True:
        # 1: get current game state from env
        old_state = agent.get_state(game)

        # 2: get action from agent
        final_move = agent.get_action(old_state)

        # 3: perform move and get feedback
        reward, done, score = game.step(final_move)

        if not eval_mode:
            new_state = agent.get_state(game)
            agent.train_short_memory(old_state, final_move, reward, new_state, done)
            agent.remember(old_state, final_move, reward, new_state, done)

        # 7: check if game over
        if done:
            # reset game layout for next iteration
            game.reset()

            if not eval_mode:
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
            else:
                print(f"Evaluation Game Complete! Final Score: {score} | All-Time Record Goal: {agent.record}")

if __name__ == '__main__':
    is_eval = False
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        is_eval = True

    run_pipeline(eval_mode=is_eval)
