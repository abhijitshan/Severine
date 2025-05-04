// tunerServer.hpp
#pragma once

#include "include/tuner.hpp"
#include "include/hyperparameter.hpp"
#include "include/bayesianOptimization.hpp"
#include <crow.h>
#include <mutex>
#include <queue>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include <set>
#include "include/json.hpp"

using json = nlohmann::json;
using namespace hypertune;

// Class for managing optimization data to be sent to the dashboard
class OptimizationHistory {
public:
    void addResult(const EvaluationResult& result, int iteration, double elapsedTime) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        json resultJson;
        resultJson["iteration"] = iteration;
        resultJson["score"] = result.score;
        resultJson["elapsed_time"] = elapsedTime;
        
        json configJson;
        for (const auto& [name, value] : result.configuration) {
            if (std::holds_alternative<int>(value)) {
                configJson[name] = std::get<int>(value);
            } else if (std::holds_alternative<float>(value)) {
                configJson[name] = std::get<float>(value);
            } else if (std::holds_alternative<bool>(value)) {
                configJson[name] = std::get<bool>(value);
            } else if (std::holds_alternative<std::string>(value)) {
                configJson[name] = std::get<std::string>(value);
            }
        }
        
        resultJson["config"] = configJson;
        history_.push_back(resultJson);
        
        // Update best result if applicable
        if (bestResult_.empty() || result.score > bestResult_["score"]) {
            bestResult_ = resultJson;
        }
        
        // Notify all clients
        for (auto& client : clients_) {
            client->send_text(resultJson.dump());
        }
    }
    
    json getHistory() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return history_;
    }
    
    json getBestResult() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return bestResult_;
    }
    
    void registerClient(crow::websocket::connection* client) {
        std::lock_guard<std::mutex> lock(mutex_);
        clients_.insert(client);
        
        // Send all history to new client
        for (const auto& result : history_) {
            client->send_text(result.dump());
        }
    }
    
    void unregisterClient(crow::websocket::connection* client) {
        std::lock_guard<std::mutex> lock(mutex_);
        clients_.erase(client);
    }
    
private:
    mutable std::mutex mutex_;
    std::vector<json> history_;
    json bestResult_;
    std::set<crow::websocket::connection*> clients_;
};

// Custom observer for optimization progress
class DashboardObserver : public TunerObserver {
public:
    DashboardObserver(OptimizationHistory& history) : history_(history), startTime_(std::chrono::high_resolution_clock::now()) {}
    
    void onIterationComplete(const EvaluationResult& result, int iteration) override {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsedTime = std::chrono::duration<double>(now - startTime_).count();
        history_.addResult(result, iteration, elapsedTime);
    }
    
private:
    OptimizationHistory& history_;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime_;
};

// Server class for managing the dashboard
class TunerServer {
public:
    TunerServer(int port = 8080) : app_(), port_(port), isRunning_(false) {
        setupRoutes();
    }
    
    ~TunerServer() {
        stop();
    }
    
    void start() {
        if (isRunning_) return;
        
        isRunning_ = true;
        serverThread_ = std::thread([this]() {
            app_.port(port_).multithreaded().run();
        });
    }
    
    void stop() {
        if (!isRunning_) return;
        
        app_.stop();
        if (serverThread_.joinable()) {
            serverThread_.join();
        }
        isRunning_ = false;
    }
    
    void registerTuner(std::shared_ptr<Tuner> tuner) {
        tuner->registerObserver(std::make_shared<DashboardObserver>(history_));
        tuners_.push_back(tuner);
    }
    
private:
    void setupRoutes() {
        // Serve static files
        CROW_ROUTE(app_, "/")([this]() {
            return dashboard_html;
        });
        
        // Get optimization history
        CROW_ROUTE(app_, "/api/history")([this]() {
            return crow::response(history_.getHistory().dump());
        });
        
        // Get best result
        CROW_ROUTE(app_, "/api/best")([this]() {
            return crow::response(history_.getBestResult().dump());
        });
        
        // WebSocket for real-time updates
        CROW_WEBSOCKET_ROUTE(app_, "/ws")
            .onopen([this](crow::websocket::connection& conn) {
                history_.registerClient(&conn);
            })
            .onclose([this](crow::websocket::connection& conn, const std::string& reason, unsigned short code) {
                history_.unregisterClient(&conn);
            })
            .onmessage([](crow::websocket::connection& conn, const std::string& msg, bool is_binary) {
                // We don't expect messages from clients
            });
    }
    
    crow::SimpleApp app_;
    int port_;
    std::atomic<bool> isRunning_;
    std::thread serverThread_;
    OptimizationHistory history_;
    std::vector<std::shared_ptr<Tuner>> tuners_;
    
    // HTML content for the dashboard
    const std::string dashboard_html = R"(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HyperTune Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.1/chart.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
        h1, h2 {
            color: #333;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        .chart-container {
            height: 300px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>HyperTune Dashboard</h1>
        
        <div class="card">
            <h2>Best Result</h2>
            <div id="best-result">Loading...</div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>Score History</h2>
                <div class="chart-container">
                    <canvas id="score-chart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>Parameter Evolution</h2>
                <div class="chart-container">
                    <canvas id="params-chart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>All Results</h2>
            <div id="results-table">Loading...</div>
        </div>
    </div>
    
    <script>
        // Global variables for chart and data
        const history = [];
        let scoreChart = null;
        let paramsChart = null;
        let paramNames = [];
        
        // Initialize websocket connection
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onopen = function() {
            console.log('WebSocket connection established');
        };
        
        ws.onmessage = function(event) {
            const result = JSON.parse(event.data);
            updateDashboard(result);
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
        };
        
        // Function to update dashboard with new data
        function updateDashboard(result) {
            // Add to history
            history.push(result);
            
            // Extract parameter names if not already done
            if (paramNames.length === 0 && result.config) {
                paramNames = Object.keys(result.config);
            }
            
            // Update best result display
            updateBestResult();
            
            // Update charts
            updateCharts();
            
            // Update results table
            updateResultsTable();
        }
        
        // Update best result display
        function updateBestResult() {
            if (history.length === 0) return;
            
            // Find best result
            const best = history.reduce((prev, current) => 
                (prev.score > current.score) ? prev : current);
            
            let html = '<table>';
            html += '<tr><th>Parameter</th><th>Value</th></tr>';
            html += `<tr><td>Score</td><td>${best.score.toFixed(6)}</td></tr>`;
            html += `<tr><td>Iteration</td><td>${best.iteration}</td></tr>`;
            
            for (const [param, value] of Object.entries(best.config)) {
                html += `<tr><td>${param}</td><td>${typeof value === 'number' ? value.toFixed(6) : value}</td></tr>`;
            }
            
            html += '</table>';
            document.getElementById('best-result').innerHTML = html;
        }
        
        // Update charts
        function updateCharts() {
            // Prepare data
            const iterations = history.map(r => r.iteration);
            const scores = history.map(r => r.score);
            
            // Score chart
            if (scoreChart === null) {
                const ctx = document.getElementById('score-chart').getContext('2d');
                scoreChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: iterations,
                        datasets: [{
                            label: 'Score',
                            data: scores,
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1,
                            fill: false
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Iteration'
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Score'
                                }
                            }
                        }
                    }
                });
            } else {
                scoreChart.data.labels = iterations;
                scoreChart.data.datasets[0].data = scores;
                scoreChart.update();
            }
            
            // Parameters chart - select up to 3 numerical parameters
            const numParams = paramNames.slice(0, 3).filter(param => 
                history.length > 0 && typeof history[0].config[param] === 'number');
            
            if (numParams.length > 0) {
                const datasets = numParams.map((param, index) => {
                    const colors = ['rgb(255, 99, 132)', 'rgb(54, 162, 235)', 'rgb(255, 206, 86)'];
                    return {
                        label: param,
                        data: history.map(r => r.config[param]),
                        borderColor: colors[index % colors.length],
                        tension: 0.1,
                        fill: false
                    };
                });
                
                if (paramsChart === null) {
                    const ctx = document.getElementById('params-chart').getContext('2d');
                    paramsChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: iterations,
                            datasets: datasets
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: 'Iteration'
                                    }
                                }
                            }
                        }
                    });
                } else {
                    paramsChart.data.labels = iterations;
                    paramsChart.data.datasets = datasets;
                    paramsChart.update();
                }
            }
        }
        
        // Update results table
        function updateResultsTable() {
            if (history.length === 0) return;
            
            // Sort by iteration
            const sortedHistory = [...history].sort((a, b) => a.iteration - b.iteration);
            
            let html = '<table>';
            html += '<tr><th>Iteration</th><th>Score</th>';
            
            // Add parameter columns
            for (const param of paramNames) {
                html += `<th>${param}</th>`;
            }
            
            html += '</tr>';
            
            // Add rows
            for (const result of sortedHistory) {
                html += `<tr>`;
                html += `<td>${result.iteration}</td>`;
                html += `<td>${result.score.toFixed(6)}</td>`;
                
                for (const param of paramNames) {
                    const value = result.config[param];
                    html += `<td>${typeof value === 'number' ? value.toFixed(6) : value}</td>`;
                }
                
                html += `</tr>`;
            }
            
            html += '</table>';
            document.getElementById('results-table').innerHTML = html;
        }
        
        // Initial data load
        async function loadInitialData() {
            try {
                const response = await fetch('/api/history');
                const data = await response.json();
                
                for (const result of data) {
                    updateDashboard(result);
                }
            } catch (error) {
                console.error('Error loading initial data:', error);
            }
        }
        
        // Call initial data load
        loadInitialData();
    </script>
</body>
</html>
    )";
};