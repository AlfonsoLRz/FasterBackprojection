#pragma once

#include "../stdafx.h"
#include "cxxopts/cxxopts.hpp"
#include "../data/SceneParameters.h"
#include "DataProcessor.h"
#include "../types.h"

namespace NLOS {

    enum class LogFileFormat {
        None,
        Binary,
        Csv,
        Yaml,
        Png,
		MonoPng
    };

    class ILogContext {
    public:
        virtual bool LogRawData() const = 0;
        virtual bool LogParsedData() const = 0;
        virtual bool LogFdhData() const = 0;
        virtual bool LogRsdData() const = 0;
        virtual bool LogImageData() const = 0;

        virtual LogFileFormat ParsedLogFormat() const = 0;
        virtual LogFileFormat FdhLogFormat() const = 0;
        virtual LogFileFormat RsdLogFormat() const = 0;
        virtual LogFileFormat ImageLogFormat() const = 0;

        virtual std::ofstream& RawLogStream() = 0;
        virtual const std::string& LogDir() const = 0;

        virtual std::string MakeLogFileName(const std::string& prefix, int frameNum, LogFileFormat fmt) = 0;
    };

	inline LogFileFormat Str2Fmt(const std::string& str) {
		std::string s = str;
		std::transform(s.begin(), s.end(), s.begin(), ::tolower);
		if (s == "bin")
			return LogFileFormat::Binary;
		else if (s == "yml")
			return LogFileFormat::Yaml;
		else if (s == "csv")
			return LogFileFormat::Csv;
		else if (s == "png")
			return LogFileFormat::Png;
		else if (s == "monopng")
			return LogFileFormat::MonoPng;
		else if (s == "none" || s.size() == 0)
			return LogFileFormat::None;
		else {
			spdlog::warn(",{:<25},Unknown log file format {}.", "LogFileFormat", str);
			return LogFileFormat::None;
		}
	}

	inline const std::string& Fmt2Str(const LogFileFormat fmt) {
		static std::map<LogFileFormat, const std::string> m{
			{ LogFileFormat::Binary, "bin"  },
			{ LogFileFormat::Yaml,   "yml"  },
			{ LogFileFormat::Csv,	 "csv"  },
			{ LogFileFormat::Png,	 "png"  },
			{ LogFileFormat::MonoPng,"monopng"  },
			{ LogFileFormat::None,	 "none" }
		};
		return m[fmt];
	}
	inline const std::string& Fmt2Ext(const LogFileFormat fmt) {
		static std::map<LogFileFormat, const std::string> m{
			{ LogFileFormat::Binary, "bin"  },
			{ LogFileFormat::Yaml,   "yml"  },
			{ LogFileFormat::Csv,	 "csv"  },
			{ LogFileFormat::Png,	 "png"  },
			{ LogFileFormat::MonoPng,"png"  },
			{ LogFileFormat::None,	 "none" }
		};
		return m[fmt];
	}
}
