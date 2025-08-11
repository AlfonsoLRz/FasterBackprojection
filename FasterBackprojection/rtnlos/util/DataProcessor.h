#pragma once

#include "cxxopts/cxxopts.hpp"
#include "../data/SceneParameters.h"

namespace NLOS {
    class LogWriter;

    class DataProcessor {
    public:
        DataProcessor(const std::string& name, LogWriter& logWriter)
            : m_name(name)
            , m_stopRequested(false)
            , m_logWriter(logWriter)
        {}

        virtual void InitCmdLineOptions(cxxopts::Options& options) { }
        virtual void Initialize(const SceneParameters& sceneParameters) { }

        void Start() {
            if (!m_thread.joinable()) {
                std::thread worker(&DataProcessor::DoWork, this);
                m_thread = std::move(worker);
            }
        }

        void Stop() {
            m_stopRequested = true;
            OnStop();
            m_thread.join();
        }

    protected:
        virtual void Work() = 0;
        virtual void OnStop() = 0;

        const std::string m_name;
        std::atomic<bool> m_stopRequested;
        LogWriter& m_logWriter;
        std::thread m_thread;
    private:
        void DoWork() {
            try {
                Work();
            }
            catch (const std::exception & ex) {
            }
        }
    };
}
