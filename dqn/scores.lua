require 'initenv'
require 'NeuralQLearner'
require 'NeuralQLearnerISA'
require 'NeuralQLearnerISArs'
require 'NeuralQLearnerISANorm'
require 'NeuralQLearnerGrad'
require 'NeuralQLearnerGradNorm'
require 'NeuralQLearner_rs'
require 'NeuralQLearner_rs2'
require 'cutorch'
require 'xlua'

op = xlua.OptionParser('filters.lua -m <model.t7>')
op:option{'-m', '--model', action='store', dest='model', help='first layer visualization', default=''}
opt = op:parse()
op:summarize()

file = torch.load(opt.model)

h = file.reward_history

print(h)
