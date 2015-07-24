require 'initenv'
require 'NeuralQLearner'
require 'cutorch'

name = 'DQN3_0_1__FULL_Y.t7'
--name = 'DQN3_0_1_seaquest_FULL_Y.t7'
--name = 'DQN3_0_1_tutankham_FULL_Y.t7'

file = torch.load(name)

file.best_model:float()

torch.save(name..'_float.t7', {best_model = file.best_model})
