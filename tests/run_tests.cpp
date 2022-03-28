#include <iostream>
#include <assert.h>

#include "../include/rollout_buffer.hpp"

void test_discrete_observation_action();
void test_buffer_multiple_envs();

int main(void) {
    //test_discrete_observation_action();
    test_buffer_multiple_envs();
}

/**
 * Very simple test that covers the basics of adding/getting from the
 * rollout buffer.
 */
void test_discrete_observation_action() {
    int buffer_size = 1;
    std::unordered_map<std::string, std::vector<long int>> obs_shape({
        {"observation", {1}}
    });
    std::vector<long int> action_shape = {1};
    float gae_lambda = 0.95;
    float gamma = 0.99;
    int num_envs = 1;

    RolloutBuffer buffer = RolloutBuffer(
        buffer_size,
        obs_shape,
        action_shape,
        gae_lambda,
        gamma,
        num_envs
    );

    torch::Tensor dummy = torch::ones({1});
    std::unordered_map<std::string, torch::Tensor> obs({
        {"observation", dummy}
    });

    buffer.add(
        obs,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy
    );

    buffer.compute_returns_and_advantages(dummy, dummy);
    RolloutBufferSamples samples = buffer.permute_and_get_samples();

    assert(samples.observations["observation"][0].item<float>() == 1.0);
}

/**
 * Test the case where the buffer is used for multiple environments
 */
void test_buffer_multiple_envs() {
    int buffer_size = 1;
    std::unordered_map<std::string, std::vector<long int>> obs_shape({
        {"observation", {1}}
    });
    std::vector<long int> action_shape = {1};
    float gae_lambda = 0.95;
    float gamma = 0.99;
    int num_envs = 3;

    RolloutBuffer buffer = RolloutBuffer(
        buffer_size,
        obs_shape,
        action_shape,
        gae_lambda,
        gamma,
        num_envs
    );

    torch::Tensor dummy1 = torch::ones({3, 1});
    torch::Tensor dummy2 = torch::ones({3});
    std::unordered_map<std::string, torch::Tensor> obs({
        {"observation", dummy1}
    });

    buffer.add(
        obs,
        dummy1,
        dummy2,
        dummy2,
        dummy2,
        dummy2
    );

    buffer.compute_returns_and_advantages(dummy2, dummy2);
    RolloutBufferSamples samples = buffer.permute_and_get_samples();
    assert(samples.observations["observation"].equal(torch::ones({3, 1})));
}