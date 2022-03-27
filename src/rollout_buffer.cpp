#include "../include/rollout_buffer.hpp"

/**
 * Construct a new instance of a rollout buffer. This buffer is used to
 * store the results of sampling an environment until enough samples have
 * been collected to update the policy.
 */
RolloutBuffer::RolloutBuffer(
    int _buffer_size,
    unordered_map<std::string, std::vector<long int>> _observation_shape,
    std::vector<long int> _action_shape,
    float _gae_lambda,
    float _gamma,
    int _num_envs
) {
    buffer_size = _buffer_size;
    observation_shapes = _observation_shape;
    action_shape = _action_shape;
    gae_lambda = _gae_lambda;
    gamma = _gamma;
    num_envs = _num_envs;

    reset();
}

/**
 * Reset the current size of the buffer to zero. Also clear all the
 * data tensors by setting them to zero tensors - this might be
 * unnecessary, and might be changed later to only be done when the
 * buffer is initialized.
 */
void RolloutBuffer::reset() {
    count = 0;

    // This forms the first two components of the dimensions of all of
    // the data tensors being stored. The observations and actions
    // require additional dimensions, and are constructed below.
    std::vector<long int> base_dim = {buffer_size, num_envs};

    // Initialize every observation key to a tensor of zeros of the
    // appropriate shape.
    for (auto &it: observation_shapes) {
        std::vector<long int> obs_dim;
        obs_dim.reserve(base_dim.size() + it.second.size());
        obs_dim.insert(obs_dim.end(), base_dim.begin(), base_dim.end());
        obs_dim.insert(obs_dim.end(), it.second.begin(), it.second.end());
        observations[it.first] = torch::zeros(obs_dim);
    }

    // Initialize the actions to an appropriately sized tensor of zeros
    std::vector<long int> act_dim;
    act_dim.reserve(base_dim.size() + action_shape.size());
    act_dim.insert(act_dim.end(), base_dim.begin(), base_dim.end());
    act_dim.insert(act_dim.end(), action_shape.begin(), action_shape.end());
    actions = torch::zeros(act_dim);

    // All following data has shape base_dim
    rewards        = torch::zeros(base_dim);
    returns        = torch::zeros(base_dim);
    episode_starts = torch::zeros(base_dim);
    values         = torch::zeros(base_dim);
    log_probs      = torch::zeros(base_dim);
    advantages     = torch::zeros(base_dim);
}

void RolloutBuffer::add(
    unordered_map<std::string, torch::Tensor> observations,
    torch::Tensor actions,
    torch::Tensor rewards,
    torch::Tensor episode_starts,
    torch::Tensor values,
    torch::Tensor log_probs
) {
    
}

RolloutBufferSamples RolloutBuffer::permute_and_get_samples() {
    
}

void RolloutBuffer::compute_returns_and_advantages(
    torch::Tensor last_values, 
    torch::Tensor dones
) {

}