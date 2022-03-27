#include <assert.h>

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
    unordered_map<std::string, torch::Tensor> _observations,
    torch::Tensor _actions,
    torch::Tensor _rewards,
    torch::Tensor _episode_starts,
    torch::Tensor _values,
    torch::Tensor _log_probs
) {
    for (auto &it: _observations) {
        observations[it.first][count] = it.second.clone();
    }

    actions[count]        = _actions.clone();
    rewards[count]        = _rewards.clone();
    episode_starts[count] = _episode_starts.clone();
    values[count]         = _values.clone();
    log_probs[count]      = _log_probs.clone();

    count += 1;
}

/**
 * Obtain the samples in this buffer as a RolloutBufferSamples object.
 * Samples will be randomly permuted.
 */
RolloutBufferSamples RolloutBuffer::permute_and_get_samples() {
    // This function should only be called when the buffer is full
    assert(count == buffer_size);

    torch::Tensor perm_indices = torch::randperm(buffer_size * num_envs);

    for (auto &it: observation_shapes) {
        observations[it.first] = 
            swap_and_flatten(observations[it.first]).index({perm_indices});
    }

    actions = swap_and_flatten(actions).index({perm_indices});
    returns = swap_and_flatten(returns).index({perm_indices});
    values = swap_and_flatten(values).index({perm_indices});
    log_probs = swap_and_flatten(log_probs).index({perm_indices});
    advantages = swap_and_flatten(advantages).index({perm_indices});

    RolloutBufferSamples samples = RolloutBufferSamples {
        observations,
        actions,
        returns,
        values,
        log_probs,
        advantages
    };

    return samples;
}

/**
 * Given a tensor with dimension (buffer_size, num_envs, ...), swaps the first
 * two axes and reshapes it into (buffer_size * num_envs, ...) then returns the
 * result. This maintains the original ordering.Does not modify the original 
 * tensor.
 */
torch::Tensor RolloutBuffer::swap_and_flatten(torch::Tensor tensor) {
    assert(tensor.dim() >= 3);

    std::vector<long int> swap_perm = {1, 0};

    for (int i = 2; i < tensor.dim(); i++) {
        swap_perm.push_back(i);
    }

    std::vector<long int> new_dim = {buffer_size * num_envs};
    for (int i = 2; i < tensor.dim(); i++) {
        new_dim.push_back(action_shape[tensor.sizes()[i]]);
    }

    return tensor.permute(torch::makeArrayRef(swap_perm)).reshape(
                torch::makeArrayRef(new_dim));
}

void RolloutBuffer::compute_returns_and_advantages(
    torch::Tensor last_values, 
    torch::Tensor dones
) {

}