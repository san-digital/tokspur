#pragma once

#include "tokspur/gguf.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <span>
#include <unistd.h>
#include <vector>

namespace tokspur {

// Private diagnostic, compiled in only with DFLASH2_TAP_DUMP=1. It writes the target taps the DFlash2 drafter is fed, with the tokens they belong to, so that a drafter can be trained offline from exactly what the runtime gives it.
// The file taps.bin is a sequence of records. A record is eight little-endian uint32 (magic "TKTP", version in the low half and source in the high half, first position, rows, width, element type 1 for IEEE half, record ordinal, zero), then `rows` token ids as uint32, then rows * width taps as IEEE half floats.
// A record whose first position is not the end of the previous record replaces everything from that position on: the worker rewound or began another request.
class DFlash2TapDump {
public:
    enum class Source : uint32_t { prefill = 0, step = 1, verification = 2 };

    explicit DFlash2TapDump(const std::filesystem::path& directory) {
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        // Exclusive create: an earlier dump is never replaced or appended to.
        const int descriptor = ::open((directory / "taps.bin").c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (descriptor < 0) throw FormatError("cannot create the DFlash2 tap dump file; the directory must not hold an earlier dump");
        file_ = ::fdopen(descriptor, "wb");
        if (!file_) {
            ::close(descriptor);
            throw FormatError("cannot open the DFlash2 tap dump file for writing");
        }
    }
    ~DFlash2TapDump() {
        if (file_) std::fclose(file_);
    }
    DFlash2TapDump(const DFlash2TapDump&) = delete;
    DFlash2TapDump& operator=(const DFlash2TapDump&) = delete;

    void write(Source source, uint32_t first_position, std::span<const uint32_t> tokens, std::span<const float> taps, uint32_t width) {
        if (tokens.empty() || !width || taps.size() != uint64_t(tokens.size()) * width) throw FormatError("DFlash2 tap dump rows differ from their tokens");
        halves_.resize(taps.size());
        for (size_t i = 0; i < taps.size(); ++i) {
            const _Float16 half = static_cast<_Float16>(taps[i]);
            // A tap that half precision cannot hold would train the drafter on a value the runtime never produces.
            if (!std::isfinite(static_cast<float>(half))) throw FormatError("DFlash2 tap does not fit half precision");
            std::memcpy(&halves_[i], &half, sizeof(uint16_t));
        }
        const uint32_t header[8] = {0x50544b54u, 1u | (uint32_t(source) << 16), first_position, uint32_t(tokens.size()), width, 1u, records_, 0u};
        if (std::fwrite(header, sizeof(header), 1, file_) != 1 ||
            std::fwrite(tokens.data(), sizeof(uint32_t), tokens.size(), file_) != tokens.size() ||
            std::fwrite(halves_.data(), sizeof(uint16_t), halves_.size(), file_) != halves_.size() ||
            std::fflush(file_) != 0) throw FormatError("cannot write a DFlash2 tap dump record");
        ++records_;
        rows_ += tokens.size();
    }

    [[nodiscard]] uint32_t records() const noexcept { return records_; }
    [[nodiscard]] uint64_t rows() const noexcept { return rows_; }

private:
    std::FILE* file_ = nullptr;
    std::vector<uint16_t> halves_;
    uint32_t records_ = 0;
    uint64_t rows_ = 0;
};

} // namespace tokspur
