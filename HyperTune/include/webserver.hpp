#pragma once
#include "crow.h"
#include <memory>
#include <mutex>
#include <vector>

namespace hypertune {

class WebServer {
public:
    WebServer(int port = 8080);
    void start();
    void stop();

    void updateProgress(const std::string& status, double score);

private:
    crow::SimpleApp app;
    int port_;
    std::mutex dataMutex_;
    std::vector<std::pair<std::string, double>> optimizationHistory_;
    std::vector<crow::websocket::connection*> clients_;  // Track WebSocket clients

    void setupRoutes();
    crow::json::wvalue getOptimizationStatus();  // Removed const qualifier
};

} // namespace hypertune