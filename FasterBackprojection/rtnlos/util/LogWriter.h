#pragma once

#include <thread>
#include <atomic>
#include <thread>
#include <fstream>
#include "cxxopts/cxxopts.hpp"
#include "../data/SceneParameters.h"
#include "DataProcessor.h"
#include "../types.h"
#include "ILogContext.h"

namespace NLOS {

    class LogWriter : public DataProcessor, ILogContext {
    public:
        LogWriter(PipelineDataQueue& incoming)
            : DataProcessor("LogWriter", *this)
            , m_incoming(incoming)
            , m_logRaw(false)
            , m_logDir("Log")
            , m_fmtPar(LogFileFormat::None)
            , m_fmtFdh(LogFileFormat::None)
            , m_fmtRsd(LogFileFormat::None)
            , m_fmtImg(LogFileFormat::None)
        {}

        virtual void PushLog(PipelineDataPtr data) { m_incoming.Push(data);  }

        virtual void InitCmdLineOptions(cxxopts::Options& options);
        virtual void Initialize(const SceneParameters& sceneParameters);

        virtual bool LogRawData() const { return m_logRaw; }
        virtual bool LogParsedData() const { return m_fmtPar != LogFileFormat::None; }
        virtual bool LogFdhData() const { return m_fmtFdh != LogFileFormat::None; }
        virtual bool LogRsdData() const { return m_fmtRsd != LogFileFormat::None; }
        virtual bool LogImageData() const { return m_fmtImg != LogFileFormat::None; }
        virtual LogFileFormat ParsedLogFormat() const { return m_fmtPar; }
        virtual LogFileFormat FdhLogFormat() const { return m_fmtFdh; }
        virtual LogFileFormat RsdLogFormat() const { return m_fmtRsd; }
        virtual LogFileFormat ImageLogFormat() const { return m_fmtImg; }

        virtual std::ofstream& RawLogStream() { return m_rawLogStream; }
        virtual const std::string& LogDir() const { return m_logDir; }

        virtual std::string MakeLogFileName(const std::string& prefix, int frameNum, LogFileFormat fmt) {
            return fmt::format("{}/{}_{:04}.{}", LogDir(), prefix, frameNum, Fmt2Ext(fmt));
        }

    protected:
        PipelineDataQueue& m_incoming;
        LogFileFormat m_format;

        std::string m_logDir;
        bool m_logRaw;
        std::string m_logParOpt;
        std::string m_logFdhOpt;
        std::string m_logRsdOpt;
        std::string m_logImgOpt;

        LogFileFormat m_fmtPar;
        LogFileFormat m_fmtFdh;
        LogFileFormat m_fmtRsd;
        LogFileFormat m_fmtImg;

        std::ofstream m_rawLogStream;

        virtual void Work();
        virtual void OnStop();

        bool PrepRawFile();
    };
}