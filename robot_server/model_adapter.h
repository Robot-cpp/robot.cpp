#ifndef ROBOT_SERVER_MODEL_ADAPTER_H
#define ROBOT_SERVER_MODEL_ADAPTER_H

#include "smolvla_protocol.h"

#include <memory>
#include <string>

namespace robotcpp {
class Model;
}

namespace robot_server {

class model_adapter {
public:
    explicit model_adapter(std::unique_ptr<robotcpp::Model> model);
    ~model_adapter();

    model_adapter(const model_adapter &) = delete;
    model_adapter & operator=(const model_adapter &) = delete;

    bool predict(
        const smolvla::protocol::predict_request & req,
        smolvla::protocol::predict_response & resp,
        std::string & error);
    void reset();
    const char * name() const;
    void set_task(std::string task);

private:
    std::unique_ptr<robotcpp::Model> model_;
    std::string task_;
};

} // namespace robot_server

#endif // ROBOT_SERVER_MODEL_ADAPTER_H
