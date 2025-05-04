#include "webserver.hpp"
#include <iostream>
#include <algorithm>

namespace hypertune {

WebServer::WebServer(int port) : port_(port) {
    setupRoutes();
}

void WebServer::setupRoutes() {
    CROW_ROUTE(app, "/static/<path>")
    ([](const std::string& path) {
        return crow::response(200, path);
    });

    CROW_ROUTE(app, "/")
    ([]() {
        return "HyperTune Dashboard - Coming Soon";
    });

    CROW_ROUTE(app, "/api/status")
    ([this]() {
        return getOptimizationStatus();
    });

    CROW_ROUTE(app, "/api/optimize")
    .methods("POST"_method)
    ([](const crow::request& req) {
        return crow::json::wvalue({
            {"status", "optimization started"}
        });
    });

    CROW_WEBSOCKET_ROUTE(app, "/ws")
        .onopen([this](crow::websocket::connection& conn) {
            CROW_LOG_INFO << "New WebSocket connection";
            std::lock_guard<std::mutex> lock(dataMutex_);
            clients_.push_back(&conn);
        })
        .onclose([this](crow::websocket::connection& conn, const std::string& reason, uint16_t code) {
            CROW_LOG_INFO << "WebSocket connection closed: " << reason << " with code: " << code;
            std::lock_guard<std::mutex> lock(dataMutex_);
            clients_.erase(std::remove(clients_.begin(), clients_.end(), &conn), clients_.end());
        })
        .onmessage([](crow::websocket::connection& conn, const std::string& data, bool is_binary) {
            CROW_LOG_INFO << "Received message: " << data;
        });
}

void WebServer::start() {
    std::cout << "Starting HyperTune server on port " << port_ << std::endl;
    app.port(port_)
       .multithreaded()
       .run();
}

void WebServer::stop() {
    app.stop();
}

void WebServer::updateProgress(const std::string& status, double score) {
    std::lock_guard<std::mutex> lock(dataMutex_);
    optimizationHistory_.push_back({status, score});

    crow::json::wvalue update({
        {"status", status},
        {"score", score}
    });

    std::string message = update.dump();
    for (auto* client : clients_) {
        // Different Crow versions might use different methods for sending messages
        // Try this if the previous one doesn't work
        client->send_text(message);
    }
}

crow::json::wvalue WebServer::getOptimizationStatus() {  // Removed const qualifier to allow locking the mutex
    std::lock_guard<std::mutex> lock(dataMutex_);
    std::vector<crow::json::wvalue> history;

    for (const auto& [status, score] : optimizationHistory_) {
        history.push_back({
            {"status", status},
            {"score", score}
        });
    }

    return crow::json::wvalue({
        {"history", history}
    });
}

} // namespace hypertune