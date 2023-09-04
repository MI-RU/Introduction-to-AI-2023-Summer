# PPO load and play (tf2 version)
# coded by St.Watermelon
## 학습된 신경망 파라미터를 가져와서 에이전트를 실행시키는 파일

# 필요한 패키지 임포트
import gym
import tensorflow as tf
from ppo_learn import PPOagent

def main():

    env_name = 'Pendulum-v0'
    env = gym.make(env_name)

    agent = PPOagent(env) # PPO 에이전트 객체

    agent.load_weights('./save_weights/') #신경망 파라미터 가져옴

    time = 0
    state = env.reset() # 환경을 초기화하고, 초기 상태 관측

    while True:
        env.render()
        # 행동 계산
        action = agent.actor(tf.convert_to_tensor([state], dtype=tf.float32))[0][0]
        # 환경으로부터 다음 상태, 보상 받음
        state, reward, done, _ = env.step(action)
        time += 1

        print('Time: ', time, 'Reward: ', reward)

        if done:
            break

    env.close()

if __name__=="__main__":
    main()