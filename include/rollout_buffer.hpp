#include <torch/torch.h>

#include <unordered_map>

/**
 * Represents a collection of samples to be used when updating the policy.
 */
class RolloutBufferSamples {
public:
    std::unordered_map<std::string, torch::Tensor> observations;
    torch::Tensor actions;
    torch::Tensor old_values;
    torch::Tensor old_log_prob;
    torch::Tensor advantages;
    torch::Tensor returns;
};

/**
 * Stores data that is sampled from an arbitrary number of parallel
 * environments. Should be cleared each time the policy is updated.
 */
class RolloutBuffer {
public:
    /**
     * Initialize a new rollout buffer
     */
    RolloutBuffer(
        int _buffer_size,
        std::unordered_map<std::string, std::vector<long int>> _observation_shapes,
        std::vector<long int> _action_shape,
        float _gae_lambda,
        float _gamma,
        int _num_envs
    );

    /**
     * Clear all current data to reset this buffer
     */
    void reset();

    /**
     * Add a set of samples to the current buffer. The first dimension of
     * each tensor corresponds to the specific environment being sampled.
     * In other words, the first dimension of each tensor should match the
     * number of parallel environments (num_envs).
     */
    void add(
        std::unordered_map<std::string, torch::Tensor> _observations,
        torch::Tensor _actions,
        torch::Tensor _rewards,
        torch::Tensor _episode_starts,
        torch::Tensor _values,
        torch::Tensor _log_probs
    );

    /**
     * Randomly permute the samples that are stored and return them. This
     * should only ever be called when the buffer is full.
     */
    RolloutBufferSamples permute_and_get_samples();

    /**
     * Compute the returns and advantages using generalized advantage
     * estimation. This should only be called when the buffer is full.
     */
    void compute_returns_and_advantages(
        torch::Tensor _last_values, 
        torch::Tensor dones
    );
private:
    // Maximum size of the buffer
    int buffer_size;

    // Observation space for the environment being sampled. Stored
    // as a hashmap to accomodate multiple components.
    std::unordered_map<std::string, std::vector<long int>> observation_shapes;

    // Action space of the environment being sampled.
    std::vector<long int> action_shape;

    // Lambda value for generalized advantage estimation
    float gae_lambda;

    // Discount factor
    float gamma;

    // Number of parallel environments being sampled. This affects the
    // shape of the internal tensors storing the data.
    int num_envs;

    // Current size of the buffer
    int count;

    /* Internally, all data is stored using torch Tensors. Because
     * observations often have more than one element (e.g. an action mask
     * along with the normal observation), they are stored within a
     * hashmap where each key corresponds to the appropriate component.
     */
    std::unordered_map<std::string, torch::Tensor> observations;
    torch::Tensor actions;
    torch::Tensor rewards;
    torch::Tensor returns;
    torch::Tensor episode_starts;
    torch::Tensor values;
    torch::Tensor log_probs;
    torch::Tensor advantages;

    torch::Tensor swap_and_flatten(torch::Tensor tensor);
};