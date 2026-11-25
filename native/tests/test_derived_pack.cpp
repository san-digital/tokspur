#include "tokspur/derived_pack.hpp"
#include "tokspur/planar_pack_v3.hpp"
#include "tokspur/expert_lanes.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"
#include "grouped_mm_fixture.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <vector>

using namespace tokspur;
using namespace tokspur::derived_pack;

namespace {
using Bytes = std::vector<std::byte>;
size_t checks = 0;

void require(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename F>
void refuses(F&& func, const char* message) {
    ++checks;
    bool caught = false;
    try {
        func();
    } catch (const FormatError&) {
        caught = true;
    } catch (const std::exception&) {
        caught = true;
    }
    if (!caught) {
        throw std::runtime_error(message);
    }
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
    const uint64_t rem = value % alignment;
    return rem == 0 ? value : value + (alignment - rem);
}

void push_u(Bytes& out, uint64_t value, size_t width) {
    for (size_t i = 0; i < width; ++i) {
        out.push_back(std::byte((value >> (8 * i)) & 0xff));
    }
}

void push_str(Bytes& out, std::string_view value) {
    push_u(out, value.size(), 8);
    for (unsigned char c : value) {
        out.push_back(std::byte(c));
    }
}

void push_meta_str(Bytes& out, std::string_view key, std::string_view value) {
    push_str(out, key);
    push_u(out, 8, 4);
    push_str(out, value);
}

void push_meta_u16(Bytes& out, std::string_view key, uint16_t value) {
    push_str(out, key);
    push_u(out, 2, 4);
    push_u(out, value, 2);
}

void push_meta_u32(Bytes& out, std::string_view key, uint32_t value) {
    push_str(out, key);
    push_u(out, 4, 4);
    push_u(out, value, 4);
}

struct SyntheticTensorSpec {
    std::string name;
    uint32_t type;
    std::vector<uint64_t> dimensions;
    Bytes payload;
};

Bytes make_valid_q8_rows(uint32_t columns, uint32_t rows, uint8_t seed) {
    const uint32_t blocks = columns / 32;
    Bytes out(size_t(rows) * blocks * 34);
    size_t pos = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t b = 0; b < blocks; ++b) {
            // Finite F16 scale 1.0 (0x3c00) varied slightly in low mantissa bits so exponent != 31.
            const uint16_t scale = uint16_t(0x3800 | ((r + b + seed) & 0x00ff));
            out[pos++] = std::byte(scale & 0xff);
            out[pos++] = std::byte((scale >> 8) & 0xff);
            for (uint32_t q = 0; q < 32; ++q) {
                out[pos++] = std::byte((r * 13 + b * 7 + q * 3 + seed) & 0xff);
            }
        }
    }
    return out;
}

Bytes make_valid_q51_rows(uint32_t columns, uint32_t rows, uint8_t seed) {
    const uint32_t blocks = columns / 32;
    Bytes out(size_t(rows) * blocks * 24);
    size_t pos = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t b = 0; b < blocks; ++b) {
            const uint16_t d = uint16_t(0x3800 | ((r + b + seed) & 0x00ff));
            const uint16_t m = uint16_t(0x3400 | ((r * 3 + b + seed) & 0x00ff));
            out[pos++] = std::byte(d & 0xff);
            out[pos++] = std::byte((d >> 8) & 0xff);
            out[pos++] = std::byte(m & 0xff);
            out[pos++] = std::byte((m >> 8) & 0xff);
            for (uint32_t h = 0; h < 4; ++h) {
                out[pos++] = std::byte((r + b + h + seed) & 0xff);
            }
            for (uint32_t q = 0; q < 16; ++q) {
                out[pos++] = std::byte((r * 17 + b * 5 + q + seed) & 0xff);
            }
        }
    }
    return out;
}

Bytes make_f32_bytes(size_t elements, uint8_t seed) {
    Bytes out(elements * 4);
    for (size_t i = 0; i < elements; ++i) {
        const float value = float(int((i + seed) % 97) - 48) * 0.125f;
        std::memcpy(out.data() + i * 4, &value, 4);
    }
    return out;
}

void write_file_bytes(const std::filesystem::path& path, const Bytes& bytes, mode_t mode = 0600) {
    const int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC | O_NOFOLLOW, mode);
    if (fd < 0) throw std::runtime_error("failed to create synthetic GGUF file");
    for (size_t at = 0; at < bytes.size();) {
        const auto n = ::write(fd, bytes.data() + at, bytes.size() - at);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            ::close(fd);
            throw std::runtime_error("failed to write synthetic GGUF file");
        }
        at += size_t(n);
    }
    ::close(fd);
}

void write_gguf_shard(
    const std::filesystem::path& path,
    std::string_view architecture,
    const std::vector<SyntheticTensorSpec>& specs,
    uint16_t split_no = 0,
    uint16_t split_count = 1,
    uint32_t split_tensors_count = 0,
    std::string_view source_revision = {}
) {
    constexpr uint64_t alignment = 32;
    Bytes header;
    push_u(header, 0x46554747, 4);
    push_u(header, 3, 4);
    push_u(header, specs.size(), 8);
    const uint64_t meta_count = (split_count > 1 ? 5 : 2) + (source_revision.empty() ? 0 : 1);
    push_u(header, meta_count, 8);
    push_meta_str(header, "general.architecture", architecture);
    push_meta_u32(header, "general.alignment", uint32_t(alignment));
    if (split_count > 1) {
        push_meta_u16(header, "split.no", split_no);
        push_meta_u16(header, "split.count", split_count);
        push_meta_u32(header, "split.tensors.count", split_tensors_count);
    }
    if (!source_revision.empty()) {
        push_meta_str(header, "tokspur.source.revision", source_revision);
    }

    std::vector<uint64_t> offsets;
    uint64_t rel = 0;
    for (const auto& s : specs) {
        rel = align_up(rel, alignment);
        offsets.push_back(rel);
        rel += s.payload.size();
    }
    for (size_t i = 0; i < specs.size(); ++i) {
        push_str(header, specs[i].name);
        push_u(header, specs[i].dimensions.size(), 4);
        for (uint64_t d : specs[i].dimensions) push_u(header, d, 8);
        push_u(header, specs[i].type, 4);
        push_u(header, offsets[i], 8);
    }
    const uint64_t data_offset = align_up(header.size(), alignment);
    header.resize(size_t(data_offset), std::byte(0));
    Bytes file = std::move(header);
    file.resize(size_t(data_offset + rel), std::byte(0));
    for (size_t i = 0; i < specs.size(); ++i) {
        std::memcpy(
            file.data() + size_t(data_offset + offsets[i]),
            specs[i].payload.data(),
            specs[i].payload.size()
        );
    }
    write_file_bytes(path, file);
}

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        const auto base = std::filesystem::canonical(std::filesystem::temp_directory_path());
        path = base /
               ("tokspur_test_derived_pack_" + std::to_string(::getpid()) + "_" + std::to_string(arc4random()));
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(
                 path, std::filesystem::directory_options::skip_permission_denied)) {
            if (!entry.is_symlink()) {
                ::chmod(entry.path().c_str(), entry.is_directory() ? 0700 : 0600);
            }
        }
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

size_t count_entries(const std::filesystem::path& dir) {
    size_t n = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        static_cast<void>(entry);
        ++n;
    }
    return n;
}

int run_cli_verify(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: test-derived_pack --verify-pack <pack> <source...> [--mtp <mtp...>]" << std::endl;
        return 2;
    }
    const std::filesystem::path pack_path = argv[2];
    std::vector<std::filesystem::path> source_paths;
    std::vector<std::filesystem::path> mtp_paths;
    bool in_mtp = false;
    for (int i = 3; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--mtp") {
            in_mtp = true;
            continue;
        }
        if (in_mtp) {
            mtp_paths.emplace_back(arg);
        } else {
            source_paths.emplace_back(arg);
        }
    }
    if (source_paths.empty()) {
        std::cerr << "missing source GGUF paths for --verify-pack" << std::endl;
        return 2;
    }
    try {
        auto source = std::make_shared<const ModelFiles>(source_paths);
        std::shared_ptr<const ModelFiles> mtp;
        if (!mtp_paths.empty()) {
            mtp = std::make_shared<const ModelFiles>(mtp_paths);
        }
        const auto single = adopt_derived_pack(source, mtp, pack_path, default_chunk_bytes, 1, TrustMode::full, false);
        const auto multi = adopt_derived_pack(source, mtp, pack_path, default_chunk_bytes, 2, TrustMode::full, false);
        require(single.manifest_sha256 == multi.manifest_sha256, "manifest digest mismatch across worker counts");
        require(single.model->tensors().size() == single.plan.tensors.size(), "adopted model tensor count mismatch");
        for (const auto& t : single.plan.tensors) {
            require(t.source_sha256.size() == 64 && t.packed_sha256.size() == 64, "missing tensor SHA-256 digest");
        }
        std::cout << "PASS verified pack tensors=" << single.plan.tensors.size()
                  << " file_bytes=" << single.plan.file_bytes
                  << " manifest_sha256=" << single.manifest_sha256 << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL verify-pack: " << e.what() << std::endl;
        return 1;
    }
}
// ---- Local seal: format, acceptance rule, files, adoption ----

std::string read_text(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read test file");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void put_text(const std::filesystem::path& path, std::string_view text, mode_t mode = 0444) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    Bytes bytes(text.size());
    std::memcpy(bytes.data(), text.data(), text.size());
    write_file_bytes(path, bytes, mode);
}

struct DirFd {
    int fd;
    explicit DirFd(const std::filesystem::path& dir) : fd(::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC)) {
        if (fd < 0) throw std::runtime_error("cannot open test directory");
    }
    ~DirFd() { ::close(fd); }
    DirFd(const DirFd&) = delete;
    DirFd& operator=(const DirFd&) = delete;
};

struct FileFd {
    int fd;
    explicit FileFd(const std::filesystem::path& path, int flags = O_RDONLY) : fd(::open(path.c_str(), flags | O_CLOEXEC)) {
        if (fd < 0) throw std::runtime_error("cannot open test file");
    }
    ~FileFd() { ::close(fd); }
    FileFd(const FileFd&) = delete;
    FileFd& operator=(const FileFd&) = delete;
};

PackFileIdentity identity_at(const std::filesystem::path& path) {
    const FileFd file(path);
    return pack_file_identity(file.fd);
}

// Rewrites one byte of a read-only file in place and makes it read-only again.
void flip_byte(const std::filesystem::path& path, uint64_t offset) {
    if (::chmod(path.c_str(), 0600) != 0) throw std::runtime_error("cannot unseal test pack");
    const FileFd file(path, O_RDWR);
    std::byte byte{};
    if (::pread(file.fd, &byte, 1, off_t(offset)) != 1) throw std::runtime_error("cannot read test byte");
    byte ^= std::byte{1};
    if (::pwrite(file.fd, &byte, 1, off_t(offset)) != 1 || ::fchmod(file.fd, 0444) != 0) {
        throw std::runtime_error("cannot alter test byte");
    }
}

void set_mtime(const std::filesystem::path& path, const PackFileIdentity& to) {
    const struct timespec times[2] = {{0, UTIME_OMIT}, {time_t(to.mtime_seconds), long(to.mtime_nanoseconds)}};
    if (::utimensat(AT_FDCWD, path.c_str(), times, 0) != 0) throw std::runtime_error("cannot set test mtime");
}

constexpr SealScope scope_none{}, scope_target{true, false}, scope_mtp{false, true}, scope_both{true, true};
static_assert(scope_both.covers(scope_both) && !scope_target.covers(scope_both) && scope_none.covers(scope_none) &&
              (scope_target | scope_mtp) == scope_both && (scope_none | scope_none) == scope_none);

// Every entry of a directory with what a write to it would change.
std::string directory_state(const std::filesystem::path& dir) {
    std::vector<std::string> entries;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        struct stat st{};
        if (::lstat(entry.path().c_str(), &st) != 0) throw std::runtime_error("cannot stat test entry");
        entries.push_back(entry.path().filename().string() + ":" + std::to_string(st.st_ino) + ":" + std::to_string(st.st_mode) +
                          ":" + std::to_string(st.st_size) + ":" + std::to_string(st.st_mtimespec.tv_sec) + "." +
                          std::to_string(st.st_mtimespec.tv_nsec) + ":" + std::to_string(st.st_ctimespec.tv_sec) + "." +
                          std::to_string(st.st_ctimespec.tv_nsec));
    }
    std::sort(entries.begin(), entries.end());
    std::string out;
    for (const auto& entry : entries) out += entry + "\n";
    return out;
}

struct SealChange {
    const char* field;
    void (*apply)(LocalSeal&);
};
// Every field a seal carries, each changed alone.
const std::array<SealChange, 9> seal_changes{{
    {"manifest", [](LocalSeal& s) { s.manifest_sha256[0] = s.manifest_sha256[0] == '0' ? '1' : '0'; }},
    {"inode", [](LocalSeal& s) { ++s.pack.inode; }},
    {"size", [](LocalSeal& s) { ++s.pack.size; }},
    {"mtime seconds", [](LocalSeal& s) { --s.pack.mtime_seconds; }},
    {"mtime nanoseconds", [](LocalSeal& s) { s.pack.mtime_nanoseconds ^= 1; }},
    {"ctime seconds", [](LocalSeal& s) { --s.pack.ctime_seconds; }},
    {"ctime nanoseconds", [](LocalSeal& s) { s.pack.ctime_nanoseconds ^= 1; }},
    {"birth time seconds", [](LocalSeal& s) { --s.pack.birthtime_seconds; }},
    {"birth time nanoseconds", [](LocalSeal& s) { s.pack.birthtime_nanoseconds ^= 1; }},
}};

std::string replaced(std::string text, std::string_view from, std::string_view to) {
    const auto at = text.find(from);
    if (at == std::string::npos) throw std::runtime_error("test seal text lacks the part to replace");
    return text.replace(at, from.size(), to);
}

void test_local_seal_format() {
    const std::string manifest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const LocalSeal seal{manifest, {556103930, 85889703936, 1794827906, 979446023, 1794827907, 62585214, 1794827906, 5}, scope_both};
    const auto text = encode_local_seal(seal);
    const std::string identity_lines = "inode=556103930\nsize=85889703936\nmtime=1794827906.979446023\n"
                                       "ctime=1794827907.062585214\nbirthtime=1794827906.000000005\n";
    require(text == "tokspur-local-seal-v2\nmanifest_sha256=" + manifest + "\ncompared=target+mtp\n" + identity_lines,
            "seal encoding is the documented eight lines");
    const auto parsed = parse_local_seal(text);
    require(parsed.form == LocalSealForm::current && parsed.seal == seal, "seal survives a round trip");
    const std::array<std::pair<SealScope, const char*>, 4> scopes{{
        {scope_none, "none"}, {scope_target, "target"}, {scope_mtp, "mtp"}, {scope_both, "target+mtp"}}};
    for (const auto& [scope, spelling] : scopes) {
        auto scoped = seal;
        scoped.compared = scope;
        const auto encoded = encode_local_seal(scoped);
        const auto back = parse_local_seal(encoded);
        require(encoded == replaced(text, "compared=target+mtp\n", std::string("compared=") + spelling + "\n") &&
                back.form == LocalSealForm::current && back.seal == scoped && back.seal.compared == scope,
                "each scope has one spelling and survives a round trip");
    }
    constexpr auto u64 = std::numeric_limits<uint64_t>::max();
    constexpr auto i64 = std::numeric_limits<int64_t>::max();
    for (const auto& pack : {PackFileIdentity{}, PackFileIdentity{u64, u64, i64, 999999999, std::numeric_limits<int64_t>::min(), 0, -1, 1}}) {
        const LocalSeal extreme{manifest, pack, scope_mtp};
        const auto encoded = encode_local_seal(extreme);
        const auto back = parse_local_seal(encoded);
        require(encoded.size() <= max_local_seal_bytes && back.form == LocalSealForm::current && back.seal == extreme,
                "extreme identity survives a round trip");
    }

    // The earlier form is recognised, so it can be replaced, but it binds no file and never vouches.
    for (const auto* tail : {"", "\n", "\r\n", " \n\n"}) {
        const auto legacy = parse_local_seal(manifest + tail);
        require(legacy.form == LocalSealForm::legacy && legacy.seal.manifest_sha256 == manifest &&
                legacy.seal.pack == PackFileIdentity{}, "manifest-only seal is the earlier form, not current");
        require(!local_seal_vouches(manifest + tail, manifest, seal.pack, scope_none) &&
                !local_seal_vouches(manifest + tail, manifest, PackFileIdentity{}, scope_none),
                "manifest-only seal never vouches");
    }

    for (size_t length = 0; length < text.size(); ++length) {
        const auto cut = text.substr(0, length);
        require(parse_local_seal(cut).form == LocalSealForm::malformed && !local_seal_vouches(cut, manifest, seal.pack, scope_none),
                "truncated seal is not current");
    }
    std::string upper = manifest;
    upper[10] = 'A';
    std::string crlf;
    for (const char c : text) crlf += c == '\n' ? std::string("\r\n") : std::string(1, c);
    const std::vector<std::string> garbage{
        "", "garbage", std::string(1, '\0'), manifest + "x", manifest.substr(1), upper, std::string(5000, 'a'),
        text + "\n", text + " ", " " + text, text + text, crlf,
        text + std::string(max_local_seal_bytes, '\n'),
        replaced(text, "seal-v2", "seal-v1"), replaced(text, "seal-v2", "seal-v3"),
        replaced(text, manifest, upper), replaced(text, manifest, manifest.substr(1)),
        replaced(text, "inode=556103930", "inode=0556103930"), replaced(text, "inode=556103930", "inode=+556103930"),
        replaced(text, "inode=556103930", "inode=-1"), replaced(text, "inode=556103930", "inode="),
        replaced(text, "inode=556103930", "inode=18446744073709551616"),
        replaced(text, "size=", "bytes="), replaced(text, "size=85889703936\n", ""),
        replaced(text, "size=85889703936\n", "size=85889703936\nsize=85889703936\n"),
        replaced(text, "inode=556103930\nsize=85889703936\n", "size=85889703936\ninode=556103930\n"),
        replaced(text, ".979446023", ".97944602"), replaced(text, ".979446023", ".9794460230"),
        replaced(text, ".979446023", ""), replaced(text, ".979446023", ".-79446023"),
        replaced(text, "mtime=1794827906", "mtime=01794827906"), replaced(text, "mtime=1794827906", "mtime=-0"),
        replaced(text, "birthtime=1794827906.000000005\n", "birthtime=1794827906.000000005\nextra=1\n"),
        // The scope line: four spellings, once, directly after the manifest.
        "tokspur-local-seal-v2\nmanifest_sha256=" + manifest + "\n" + identity_lines,
        replaced(text, "compared=target+mtp\n", ""),
        replaced(text, "compared=target+mtp\n", "compared=target+mtp\ncompared=target+mtp\n"),
        replaced(text, "compared=target+mtp\n", "compared=target\ncompared=mtp\n"),
        replaced(text, "compared=target+mtp\ninode=556103930\n", "inode=556103930\ncompared=target+mtp\n"),
        replaced(text, "compared=target+mtp", "compared=mtp+target"), replaced(text, "compared=target+mtp", "compared=target+mtp+"),
        replaced(text, "compared=target+mtp", "compared=target,mtp"), replaced(text, "compared=target+mtp", "compared=target+target"),
        replaced(text, "compared=target+mtp", "compared=TARGET+MTP"), replaced(text, "compared=target+mtp", "compared= target"),
        replaced(text, "compared=target+mtp", "compared=target "), replaced(text, "compared=target+mtp", "compared="),
        replaced(text, "compared=target+mtp", "compared=both"), replaced(text, "compared=target+mtp", "compared=all"),
        replaced(text, "compared=target+mtp", "compared=1"), replaced(text, "compared=target+mtp", "compared"),
        replaced(text, "compared=target+mtp", "scope=target+mtp"),
    };
    for (const auto& bad : garbage) {
        require(parse_local_seal(bad).form == LocalSealForm::malformed && !local_seal_vouches(bad, manifest, seal.pack, scope_none),
                "malformed seal is not current");
    }

    // The acceptance rule: the manifest and every identity field, exactly.
    require(local_seal_vouches(text, manifest, seal.pack, scope_both), "exact seal vouches");
    for (const auto& change : seal_changes) {
        auto other = seal;
        change.apply(other);
        require(other != seal && !local_seal_vouches(encode_local_seal(other), seal.manifest_sha256, seal.pack, scope_none) &&
                !local_seal_vouches(text, other.manifest_sha256, other.pack, scope_none),
                change.field);
    }
    require(!local_seal_vouches(text, "", seal.pack, scope_none) && !local_seal_vouches(text, upper, seal.pack, scope_none),
            "malformed expected manifest never vouches");

    // The scope rule: a seal vouches only for a start with no source beyond the ones it was compared with.
    //                          needed: none   target mtp    target+mtp
    const bool expected[4][4] = {{true, false, false, false},  // compared=none
                                 {true, true, false, false},   // compared=target
                                 {true, false, true, false},   // compared=mtp
                                 {true, true, true, true}};    // compared=target+mtp
    for (size_t have = 0; have < 4; ++have) {
        auto scoped = seal;
        scoped.compared = scopes[have].first;
        for (size_t need = 0; need < 4; ++need) {
            require(local_seal_vouches(encode_local_seal(scoped), manifest, seal.pack, scopes[need].first) == expected[have][need] &&
                    scopes[have].first.covers(scopes[need].first) == expected[have][need],
                    "seal scope must cover the sources this start would compare");
            const auto both = scopes[have].first | scopes[need].first;
            require(both.covers(scopes[have].first) && both.covers(scopes[need].first) &&
                    both.target == (scopes[have].first.target || scopes[need].first.target) &&
                    both.mtp == (scopes[have].first.mtp || scopes[need].first.mtp), "scope union holds exactly both");
        }
    }
}

void test_local_seal_files(const std::filesystem::path& dir) {
    std::filesystem::create_directory(dir);
    const std::string manifest(64, 'c');
    const auto pack = dir / "p.asa.gguf", seal = dir / "p.asa.seal";
    put_text(pack, std::string(4096, 'p'));
    require(local_seal_name("p.asa.gguf") == "p.asa.seal" && local_seal_name("tokspur-pack-v3.asa.gguf") == "tokspur-pack-v3.asa.seal" &&
            local_seal_name("blob") == "blob.seal" && local_seal_name(".seal") == ".seal.seal",
            "seal name is the pack stem plus .seal");
    require(!local_seal_name("p.seal") && !local_seal_name("p.asa.seal") && !local_seal_name("blob.seal"),
            "pack that would be its own seal has no seal name");
    const DirFd d(dir);
    const FileFd pack_fd(pack);
    const auto identity = pack_file_identity(pack_fd.fd);
    struct stat pack_st{};
    require(::fstat(pack_fd.fd, &pack_st) == 0 &&
            identity == PackFileIdentity{uint64_t(pack_st.st_ino), uint64_t(pack_st.st_size),
                                         int64_t(pack_st.st_mtimespec.tv_sec), uint32_t(pack_st.st_mtimespec.tv_nsec),
                                         int64_t(pack_st.st_ctimespec.tv_sec), uint32_t(pack_st.st_ctimespec.tv_nsec),
                                         int64_t(pack_st.st_birthtimespec.tv_sec), uint32_t(pack_st.st_birthtimespec.tv_nsec)},
            "identity is fstat of the descriptor at full timestamp resolution");
    refuses([&] { static_cast<void>(pack_file_identity(d.fd)); }, "identity of a directory refused");
    const auto reading = [&](SealScope needed = scope_both) {
        return read_local_seal(d.fd, "p.asa.gguf", manifest, identity, needed);
    };
    using Entry = LocalSealReading::Entry;

    require(reading().entry == Entry::absent && !reading().vouches, "absent seal does not vouch");
    refuses([&] { publish_local_seal(d.fd, "p.asa.gguf", {"not a hash", identity, scope_both}, false); }, "malformed manifest is never sealed");
    publish_local_seal(d.fd, "p.asa.gguf", {manifest, identity, scope_both}, false);
    struct stat st{};
    require(::lstat(seal.c_str(), &st) == 0 && S_ISREG(st.st_mode) && (st.st_mode & 0777) == 0444 && st.st_nlink == 1 &&
            count_entries(dir) == 2, "published seal is a read-only single-link file and no temporary remains");
    require(read_text(seal) == encode_local_seal({manifest, identity, scope_both}), "published seal is the exact encoding");
    auto found = reading();
    require(found.entry == Entry::regular && found.trusted_file && found.form == LocalSealForm::current &&
            found.names_manifest && found.names_file && found.compared == scope_both && found.vouches,
            "published seal vouches for the file it names");
    refuses([&] { publish_local_seal(d.fd, "p.asa.gguf", {manifest, {}, scope_both}, false); }, "exclusive publication never replaces");
    require(reading().vouches && count_entries(dir) == 2, "refused publication left the seal and no temporary");

    // A sound seal file that does not match exactly.
    for (const auto& change : seal_changes) {
        LocalSeal other{manifest, identity, scope_both};
        change.apply(other);
        publish_local_seal(d.fd, "p.asa.gguf", other, true);
        found = reading();
        require(found.trusted_file && found.form == LocalSealForm::current && !found.vouches && !found.names_file &&
                found.names_manifest == (other.manifest_sha256 == manifest), change.field);
        require(!reading(scope_none).vouches, change.field);
    }
    // A sound seal for exactly this file with a narrower scope: it names the file, and vouches only for what it covers.
    for (const auto scope : {scope_none, scope_target, scope_mtp, scope_both}) {
        publish_local_seal(d.fd, "p.asa.gguf", {manifest, identity, scope}, true);
        for (const auto needed : {scope_none, scope_target, scope_mtp, scope_both}) {
            found = reading(needed);
            require(found.trusted_file && found.names_manifest && found.names_file && found.compared == scope &&
                    found.vouches == scope.covers(needed), "seal file vouches only within its scope");
        }
    }
    put_text(seal, manifest + "\n");
    found = reading();
    require(found.trusted_file && found.form == LocalSealForm::legacy && found.names_manifest && !found.vouches,
            "earlier-format seal file does not vouch");
    require(!found.names_file && found.compared == scope_none, "earlier-format seal names no file and no scope");
    put_text(seal, encode_local_seal({manifest, identity, scope_both}).substr(0, 40));
    found = reading();
    require(found.trusted_file && found.form == LocalSealForm::malformed && !found.names_manifest && !found.vouches,
            "truncated seal file does not vouch");

    // A seal file that fails the file checks is not read at all, whatever it says.
    const auto good = encode_local_seal({manifest, identity, scope_both});
    put_text(seal, good, 0644);
    found = reading();
    require(found.entry == Entry::regular && !found.trusted_file && !found.names_file && found.compared == scope_none &&
            !found.vouches && !reading(scope_none).vouches, "writable seal does not vouch");
    put_text(seal, good);
    require(reading().vouches && ::link(seal.c_str(), (dir / "alias").c_str()) == 0, "link a second name to the seal");
    require(!reading().trusted_file && !reading().vouches, "multiply linked seal does not vouch");
    require(std::filesystem::remove(dir / "alias") && reading().vouches, "single link restored");
    put_text(seal, good + std::string(max_local_seal_bytes, '\n'));
    found = reading();
    require(found.entry == Entry::other && !found.trusted_file && !found.vouches, "oversized seal does not vouch");
    put_text(seal, std::string(max_local_seal_bytes, 's'));
    require(reading().entry == Entry::regular && !reading().vouches, "largest seal-sized file is still a seal candidate");
    put_text(seal, "");
    require(!reading().trusted_file && !reading().vouches, "empty seal does not vouch");
    put_text(dir / "real", good);
    require(std::filesystem::remove(seal) && ::symlink("real", seal.c_str()) == 0, "point a symlink at a good seal");
    found = reading();
    require(found.entry == Entry::other && !found.trusted_file && !found.vouches, "symlinked seal does not vouch");

    // Refresh after a full verification of pack_fd.
    const auto refresh = [&](const PackFileIdentity& before = PackFileIdentity{}, const char* name = "p.asa.gguf",
                             SealScope verified = scope_both) {
        return refresh_local_seal(d.fd, name, pack_fd.fd, manifest, before == PackFileIdentity{} ? identity : before, verified);
    };
    require(!refresh() && std::filesystem::is_symlink(seal) && read_text(dir / "real") == good,
            "symlinked seal name is neither replaced nor followed");
    require(std::filesystem::remove(seal) && std::filesystem::create_directory(seal) && !refresh() &&
            std::filesystem::is_directory(seal) && std::filesystem::remove(seal), "directory at the seal name is left alone");
    {
        // Another file that merely has the seal name, such as a pack called p.asa.seal.
        const std::string bystander(max_local_seal_bytes + 1, 'b');
        for (const mode_t mode : {mode_t(0444), mode_t(0644)}) {
            put_text(seal, bystander, mode);
            const auto before = identity_at(seal);
            require(!refresh() && read_text(seal) == bystander && identity_at(seal) == before,
                    "file too large to be a seal is not replaced");
            refuses([&] { publish_local_seal(d.fd, "p.asa.gguf", {manifest, identity, scope_both}, false); },
                    "exclusive publication does not replace it either");
            require(read_text(seal) == bystander && identity_at(seal) == before && std::filesystem::remove(seal),
                    "bystander file is intact");
        }
    }
    for (const auto& change : seal_changes) {
        LocalSeal before{manifest, identity, scope_both};
        change.apply(before);
        if (before.pack == identity) continue;
        require(!refresh(before.pack) && !std::filesystem::exists(seal), "file that changed during verification is not sealed");
    }
    put_text(dir / "q.asa.gguf", std::string(4096, 'q'));
    require(!refresh({}, "q.asa.gguf") && !refresh({}, "missing.gguf") && !std::filesystem::exists(dir / "q.asa.seal"),
            "name that is not the verified file is not sealed");
    {
        const DirFd other(dir);
        require(::flock(other.fd, LOCK_EX | LOCK_NB) == 0 && !refresh() && !std::filesystem::exists(seal),
                "directory held by a publisher is not written");
    }
    if (::geteuid() != 0) {
        require(::chmod(dir.c_str(), 0555) == 0 && !refresh() && ::chmod(dir.c_str(), 0700) == 0 &&
                !std::filesystem::exists(seal), "unwritable directory yields no seal and no error");
    }
    require(refresh() && reading().vouches && read_text(seal) == good, "missing seal is created");
    require(::stat(seal.c_str(), &st) == 0, "stat created seal");
    const auto kept = st.st_ino;
    require(refresh() && ::stat(seal.c_str(), &st) == 0 && st.st_ino == kept, "current seal is kept, not rewritten");
    for (const auto* stale : {"garbage", "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n"}) {
        put_text(seal, stale);
        require(refresh() && reading().vouches, "stale regular seal is replaced");
    }
    put_text(seal, good, 0644);
    require(refresh() && reading().vouches && ::stat(seal.c_str(), &st) == 0 && (st.st_mode & 0777) == 0444,
            "writable seal is replaced by a read-only one");

    // Scope on refresh. A seal that names this manifest and exactly this file
    // keeps what it compared and gains what was just verified.
    const auto sealed_as = [&](SealScope scope) { return read_text(seal) == encode_local_seal({manifest, identity, scope}); };
    const auto verified = [&](SealScope scope) { return refresh({}, "p.asa.gguf", scope); };
    for (const auto have : {scope_none, scope_target, scope_mtp, scope_both}) {
        for (const auto add : {scope_none, scope_target, scope_mtp, scope_both}) {
            publish_local_seal(d.fd, "p.asa.gguf", {manifest, identity, have}, true);
            require(::stat(seal.c_str(), &st) == 0, "stat seal before refresh");
            const auto before_inode = st.st_ino;
            require(verified(add) && sealed_as(have | add) && ::stat(seal.c_str(), &st) == 0 &&
                    (st.st_ino == before_inode) == have.covers(add),
                    "seal for the same file is widened to the union, and kept when it already covers");
        }
    }
    // Any other seal contributes nothing: the new seal carries the verified scope alone.
    LocalSeal elsewhere{manifest, identity, scope_both};
    ++elsewhere.pack.inode;
    put_text(seal, encode_local_seal(elsewhere));
    require(verified(scope_mtp) && sealed_as(scope_mtp), "seal for another file identity is not widened");
    put_text(seal, encode_local_seal({std::string(64, 'd'), identity, scope_both}));
    require(verified(scope_target) && sealed_as(scope_target), "seal for another manifest is not widened");
    put_text(seal, manifest + "\n");
    require(verified(scope_none) && sealed_as(scope_none), "earlier-format seal is not widened");
    put_text(seal, good, 0644);
    require(verified(scope_target) && sealed_as(scope_target), "seal that fails the file checks is not widened");
    put_text(seal, good);
    require(::link(seal.c_str(), (dir / "alias").c_str()) == 0 && verified(scope_mtp) && sealed_as(scope_mtp) &&
            std::filesystem::remove(dir / "alias"), "multiply linked seal is not widened");
    require(verified(scope_both) && sealed_as(scope_both), "seal is back to both sources");

    // A pack whose name is its own seal name: never read as a seal, never sealed, never written.
    const auto own = dir / "own-seal";
    std::filesystem::create_directory(own);
    for (const auto* name : {"p.seal", "tokspur-pack-v2.asa.seal"}) {
        put_text(own / name, std::string(300, 's'));
        const DirFd own_dir(own);
        const FileFd own_fd(own / name);
        const auto own_identity = pack_file_identity(own_fd.fd);
        const auto state = directory_state(own);
        const auto bytes = read_text(own / name);
        const auto own_reading = read_local_seal(own_dir.fd, name, manifest, own_identity, scope_none);
        require(own_reading.entry == Entry::other && !own_reading.trusted_file && !own_reading.names_manifest &&
                !own_reading.names_file && !own_reading.vouches, "pack that is its own seal name is never read as a seal");
        for (const bool replace : {false, true}) {
            refuses([&] { publish_local_seal(own_dir.fd, name, {manifest, own_identity, scope_both}, replace); },
                    "pack that is its own seal name is never sealed");
        }
        require(!refresh_local_seal(own_dir.fd, name, own_fd.fd, manifest, own_identity, scope_both),
                "pack that is its own seal name is not refreshed");
        require(directory_state(own) == state && read_text(own / name) == bytes && pack_file_identity(own_fd.fd) == own_identity &&
                count_entries(own) == 1, "pack that is its own seal name is untouched");
        require(std::filesystem::remove(own / name), "remove own-seal fixture");
    }

    // What changes the identity of the sealed file.
    const auto vouches_for = [&](const std::filesystem::path& file) {
        return read_local_seal(d.fd, "p.asa.gguf", manifest, identity_at(file), scope_both).vouches;
    };
    require(vouches_for(pack), "reading the pack again changes nothing");
    flip_byte(pack, 100);
    auto now = identity_at(pack);
    require(now.inode == identity.inode && now.size == identity.size && now.birthtime_seconds == identity.birthtime_seconds &&
            (now.ctime_seconds != identity.ctime_seconds || now.ctime_nanoseconds != identity.ctime_nanoseconds) &&
            !vouches_for(pack), "payload rewritten in place is not vouched for");
    set_mtime(pack, identity);
    now = identity_at(pack);
    require(now.mtime_seconds == identity.mtime_seconds && now.mtime_nanoseconds == identity.mtime_nanoseconds &&
            !vouches_for(pack), "restoring the modification time does not restore the seal");
    std::filesystem::create_directory(dir / "copy");
    std::filesystem::copy_file(pack, dir / "copy" / "p.asa.gguf");
    require(identity_at(dir / "copy" / "p.asa.gguf").inode != identity.inode && !vouches_for(dir / "copy" / "p.asa.gguf"),
            "copied pack is not vouched for");
    require(refresh_local_seal(d.fd, "p.asa.gguf", pack_fd.fd, manifest, identity_at(pack), scope_both) && vouches_for(pack),
            "seal follows the file after a new verification");
    std::filesystem::rename(pack, dir / "moved.asa.gguf");
    require(!vouches_for(dir / "moved.asa.gguf"), "renamed pack is not vouched for");
}

// The smallest source and sidecar that satisfy the full v2 census: 1,222 target tensors and 32 blk.48 tensors.
struct SealedPackFixture {
    std::shared_ptr<const ModelFiles> source, changed_source, mtp, changed_mtp;
    std::filesystem::path pack, seal;
    PackPlan built;
};

SealedPackFixture build_sealed_pack_fixture(const std::filesystem::path& dir) {
    std::filesystem::create_directory(dir);
    std::vector<SyntheticTensorSpec> target{{"token_embd.weight", 0, {8}, make_f32_bytes(8, 1)}};
    for (const auto* name : {"output.weight", "output_hc_norm.weight", "blk.0.ssm_a", "blk.3.attn_q.weight", "blk.1.ple_key.weight"}) {
        target.push_back({name, 0, {8}, make_f32_bytes(8, uint8_t(target.size()))});
    }
    while (target.size() < 1 + expected_gpu_tensors - 32) {
        const auto i = target.size();
        target.push_back({"blk." + std::to_string(i % 48) + ".seal_fixture_" + std::to_string(i) + ".weight", 0, {8},
                          make_f32_bytes(8, uint8_t(i))});
    }
    std::vector<SyntheticTensorSpec> sidecar;
    for (size_t i = 0; i < 32; ++i) {
        sidecar.push_back({"blk.48.seal_fixture_" + std::to_string(i) + ".weight", 0, {8}, make_f32_bytes(8, uint8_t(i + 3))});
    }
    auto changed = target;
    changed[700].payload.back() ^= std::byte{1};
    write_gguf_shard(dir / "source.gguf", "qwen4exp", target);
    write_gguf_shard(dir / "changed-source.gguf", "qwen4exp", changed);
    write_gguf_shard(dir / "mtp.gguf", "qwen4exp", sidecar);
    auto changed_sidecar = sidecar;
    changed_sidecar[17].payload.back() ^= std::byte{1};
    write_gguf_shard(dir / "changed-mtp.gguf", "qwen4exp", changed_sidecar);
    SealedPackFixture fixture;
    fixture.source = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{dir / "source.gguf"});
    fixture.changed_source = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{dir / "changed-source.gguf"});
    fixture.mtp = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{dir / "mtp.gguf"});
    fixture.changed_mtp = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{dir / "changed-mtp.gguf"});
    std::filesystem::create_directory(dir / "derived");
    fixture.pack = dir / "derived" / "tokspur-pack-v2.asa.gguf";
    fixture.seal = dir / "derived" / "tokspur-pack-v2.asa.seal";
    fixture.built = build_derived_pack(fixture.source, fixture.mtp, fixture.pack, 1ULL << 20, 32'768);
    require(verify_gpu_pack_census(fixture.built), "seal fixture satisfies the full v2 census");
    return fixture;
}

void test_local_seal_adoption(const std::filesystem::path& dir) {
    const auto fixture = build_sealed_pack_fixture(dir);
    auto pack = fixture.pack, seal = fixture.seal;
    const auto adopt = [&](const std::shared_ptr<const ModelFiles>& origin, TrustMode mode) {
        return adopt_derived_pack(origin, fixture.mtp, pack, 32'768, 4, mode);
    };
    const auto manifest = adopt(fixture.source, TrustMode::full).manifest_sha256;
    const auto current = [&] { return LocalSeal{manifest, identity_at(pack), scope_both}; };
    // Everything observable about the seal name, to show a refused start left it alone.
    const auto seal_state = [&]() -> std::string {
        struct stat st{};
        if (::lstat(seal.c_str(), &st) != 0) return "absent";
        if (!S_ISREG(st.st_mode)) return "not a file";
        return std::to_string(st.st_ino) + ":" + std::to_string(st.st_mode) + ":" + std::to_string(st.st_nlink) + ":" + read_text(seal);
    };
    // The changed source has the same tensor list and one different payload
    // byte, so it is adopted only when the payload comparison does not run.
    const auto falls_back = [&](const char* what) {
        const auto state = seal_state();
        refuses([&] { static_cast<void>(adopt(fixture.changed_source, TrustMode::local_seal)); }, what);
        require(seal_state() == state, "failed full verification wrote no seal");
    };
    const auto accepts = [&](const char* what) {
        const auto state = seal_state();
        const auto adopted = adopt(fixture.changed_source, TrustMode::local_seal);
        require(adopted.local_seal == LocalSealUse::accepted && adopted.authentication_workers == 1 && adopted.compared == scope_none &&
                adopted.manifest_sha256 == manifest && adopted.pack_identity == identity_at(pack) && seal_state() == state, what);
    };
    const auto refreshes = [&](const char* what) {
        const auto adopted = adopt(fixture.source, TrustMode::local_seal);
        struct stat st{};
        require(adopted.local_seal == LocalSealUse::refreshed && adopted.authentication_workers == 4 && adopted.compared == scope_both &&
                adopted.pack_identity == identity_at(pack) && ::lstat(seal.c_str(), &st) == 0 && S_ISREG(st.st_mode) &&
                (st.st_mode & 0777) == 0444 && st.st_nlink == 1 && read_text(seal) == encode_local_seal(current()), what);
    };
    const auto leaves_no_seal = [&](const char* what) {
        const auto state = seal_state();
        const auto adopted = adopt(fixture.source, TrustMode::local_seal);
        require(adopted.local_seal == LocalSealUse::not_refreshed && adopted.authentication_workers == 4 &&
                adopted.compared == scope_both && seal_state() == state, what);
    };

    // Full mode neither reads nor writes a seal.
    require(adopt(fixture.source, TrustMode::full).local_seal == LocalSealUse::not_consulted &&
            adopt(fixture.source, TrustMode::full).compared == scope_both && seal_state() == "absent",
            "full mode writes no seal");
    falls_back("missing seal falls back to full verification");
    refreshes("full verification in local-seal mode leaves a current seal");
    accepts("current seal is accepted without comparing the payload");
    refuses([&] { static_cast<void>(adopt(fixture.changed_source, TrustMode::full)); }, "full mode ignores a valid seal");
    require(count_entries(pack.parent_path()) == 2, "no seal temporary remains");

    for (const auto& change : seal_changes) {
        auto other = current();
        change.apply(other);
        put_text(seal, encode_local_seal(other));
        falls_back(change.field);
        refreshes(change.field);
        accepts(change.field);
    }

    put_text(seal, "garbage\n");
    const auto garbage_state = seal_state();
    require(adopt(fixture.source, TrustMode::full).local_seal == LocalSealUse::not_consulted && seal_state() == garbage_state,
            "full mode leaves an unusable seal alone");
    falls_back("malformed seal falls back");
    refreshes("malformed seal is replaced after full verification");
    put_text(seal, manifest + "\n");
    falls_back("earlier-format seal falls back");
    refreshes("earlier-format seal is replaced after full verification");
    const auto good = encode_local_seal(current());
    put_text(seal, good.substr(0, good.size() - 1));
    falls_back("truncated seal falls back");
    refreshes("truncated seal is replaced after full verification");
    put_text(seal, good, 0644);
    falls_back("writable seal falls back");
    refreshes("writable seal is replaced after full verification");
    const auto alias = seal.parent_path() / "alias";
    require(::link(seal.c_str(), alias.c_str()) == 0, "link a second name to the seal");
    falls_back("multiply linked seal falls back");
    refreshes("multiply linked seal is replaced after full verification");
    require(read_text(alias) == good && std::filesystem::remove(alias), "other link kept its own file");
    const auto real = seal.parent_path() / "real";
    std::filesystem::rename(seal, real);
    require(::symlink("real", seal.c_str()) == 0, "point a symlink at a good seal");
    falls_back("symlinked seal falls back");
    leaves_no_seal("symlinked seal name is never replaced");
    require(std::filesystem::is_symlink(seal) && read_text(real) == good && std::filesystem::remove(seal) &&
            std::filesystem::remove(real) && std::filesystem::create_directory(seal), "replace the symlink with a directory");
    falls_back("directory at the seal name falls back");
    leaves_no_seal("directory at the seal name is never replaced");
    require(std::filesystem::remove(seal), "remove directory at the seal name");
    {
        const DirFd other(pack.parent_path());
        require(::flock(other.fd, LOCK_EX | LOCK_NB) == 0, "hold the pack directory as a publisher does");
        leaves_no_seal("directory held by a publisher is not written");
    }
    if (::geteuid() != 0) {
        require(::chmod(pack.parent_path().c_str(), 0555) == 0, "make the pack directory unwritable");
        leaves_no_seal("unwritable directory costs a full verification, not an error");
        require(::chmod(pack.parent_path().c_str(), 0700) == 0, "restore the pack directory");
    }
    refreshes("seal is written once the directory allows it");
    accepts("refreshed seal is accepted");

    // The seal records which sources the pack was compared with. Without the
    // MTP sidecar the blk.48.* tensors are checked against the manifest only,
    // so that seal must not be accepted by a start that has the sidecar.
    const auto adopt_with = [&](const std::shared_ptr<const ModelFiles>& origin, const std::shared_ptr<const ModelFiles>& sidecar,
                                TrustMode mode = TrustMode::local_seal) {
        return adopt_derived_pack(origin, sidecar, pack, 32'768, 4, mode);
    };
    const auto sealed_as = [&](SealScope scope) {
        return read_text(seal) == encode_local_seal({manifest, identity_at(pack), scope});
    };
    require(adopt_with(fixture.source, nullptr, TrustMode::full).compared == scope_target && sealed_as(scope_both),
            "full mode without the sidecar compares the target only and leaves the seal alone");
    refuses([&] { static_cast<void>(adopt_with(fixture.source, fixture.changed_mtp, TrustMode::full)); },
            "full mode compares the sidecar");
    require(std::filesystem::remove(seal), "remove the seal for the scope sequence");
    auto scoped = adopt_with(fixture.source, nullptr);
    require(scoped.local_seal == LocalSealUse::refreshed && scoped.authentication_workers == 4 && scoped.compared == scope_target &&
            sealed_as(scope_target), "start without the sidecar seals the target comparison only");
    scoped = adopt_with(fixture.changed_source, nullptr);
    require(scoped.local_seal == LocalSealUse::accepted && scoped.authentication_workers == 1 && scoped.compared == scope_none &&
            sealed_as(scope_target), "start without the sidecar accepts the target-only seal");
    auto scope_state = seal_state();
    refuses([&] { static_cast<void>(adopt_with(fixture.source, fixture.changed_mtp)); },
            "start with the sidecar does not accept a target-only seal: the sidecar is compared");
    refuses([&] { static_cast<void>(adopt_with(fixture.changed_source, fixture.mtp)); },
            "start with the sidecar does not accept a target-only seal: the target is compared again");
    require(seal_state() == scope_state, "refused starts left the target-only seal alone");
    scoped = adopt_with(fixture.source, fixture.mtp);
    require(scoped.local_seal == LocalSealUse::refreshed && scoped.authentication_workers == 4 && scoped.compared == scope_both &&
            sealed_as(scope_both), "start with the sidecar widens the seal to both sources");
    scope_state = seal_state();
    for (const auto& origin : {fixture.source, fixture.changed_source}) {
        for (const auto& sidecar : {fixture.mtp, fixture.changed_mtp, std::shared_ptr<const ModelFiles>{}}) {
            scoped = adopt_with(origin, sidecar);
            require(scoped.local_seal == LocalSealUse::accepted && scoped.authentication_workers == 1 &&
                    scoped.compared == scope_none && seal_state() == scope_state,
                    "seal for both sources is accepted with or without the sidecar, and neither source is read");
        }
    }
    // A narrower seal for another identity of the file contributes nothing to the new one.
    LocalSeal elsewhere{manifest, identity_at(pack), scope_both};
    ++elsewhere.pack.inode;
    put_text(seal, encode_local_seal(elsewhere));
    scoped = adopt_with(fixture.source, nullptr);
    require(scoped.local_seal == LocalSealUse::refreshed && scoped.compared == scope_target && sealed_as(scope_target),
            "seal for another file identity is replaced by the verified scope alone");
    refreshes("seal is widened back to both sources");

    // A pack whose own name ends in .seal has no seal name. It is fully
    // verified on every start and nothing in its directory is written.
    {
        const auto own = dir / "own-seal";
        std::filesystem::create_directory(own);
        const auto named = own / "tokspur-pack-v2.seal";
        std::filesystem::copy_file(pack, named);
        require(::chmod(named.c_str(), 0444) == 0, "make the .seal-named pack read-only");
        const auto bytes = read_text(named);
        const auto identity = identity_at(named);
        const auto state = directory_state(own);
        require(bytes == read_text(pack) && bytes.size() == fixture.built.file_bytes, "the .seal-named pack is a copy of the fixture pack");
        for (int start = 0; start < 2; ++start) {
            refuses([&] { static_cast<void>(adopt_derived_pack(fixture.changed_source, fixture.mtp, named, 32'768, 4, TrustMode::local_seal)); },
                    "the .seal-named pack is fully verified on every start");
            const auto adopted = adopt_derived_pack(fixture.source, fixture.mtp, named, 32'768, 4, TrustMode::local_seal);
            struct stat st{};
            require(adopted.local_seal == LocalSealUse::not_refreshed && adopted.authentication_workers == 4 &&
                    adopted.compared == scope_both && adopted.manifest_sha256 == manifest && adopted.pack_identity == identity,
                    "the .seal-named pack is adopted after full verification with no seal");
            require(::lstat(named.c_str(), &st) == 0 && S_ISREG(st.st_mode) && (st.st_mode & 0777) == 0444 &&
                    identity_at(named) == identity && read_text(named) == bytes && directory_state(own) == state &&
                    count_entries(own) == 1, "the .seal-named pack is byte-identical and nothing was added beside it");
        }
    }

    // Payload bytes altered in place: the seal no longer matches, and full verification refuses the pack.
    const auto sealed = current();
    const auto sealed_state = seal_state();
    const auto payload = fixture.built.data_offset + fixture.built.tensors[5].offset;
    flip_byte(pack, payload);
    auto now = identity_at(pack);
    require(now.inode == sealed.pack.inode && now.size == sealed.pack.size && now != sealed.pack,
            "in-place write keeps the inode and size and changes the times");
    refuses([&] { static_cast<void>(adopt(fixture.source, TrustMode::local_seal)); },
            "altered payload falls back and fails full verification");
    set_mtime(pack, sealed.pack);
    now = identity_at(pack);
    require(now.mtime_seconds == sealed.pack.mtime_seconds && now.mtime_nanoseconds == sealed.pack.mtime_nanoseconds &&
            now != sealed.pack, "restored modification time leaves the change time different");
    refuses([&] { static_cast<void>(adopt(fixture.source, TrustMode::local_seal)); },
            "altered payload with a restored modification time still fails");
    require(seal_state() == sealed_state, "no seal is issued for altered bytes");
    flip_byte(pack, payload);
    falls_back("restored payload is a changed file: it falls back");
    refreshes("restored payload is sealed again after full verification");

    // The pack and its seal copied elsewhere: a new file, so one full verification.
    const auto original_pack = pack, original_seal = seal;
    const auto copies = dir / "copied";
    std::filesystem::create_directory(copies);
    std::filesystem::copy_file(pack, copies / pack.filename());
    put_text(copies / seal.filename(), read_text(seal));
    set_mtime(copies / pack.filename(), identity_at(pack));
    pack = copies / pack.filename();
    seal = copies / seal.filename();
    require(::chmod(pack.c_str(), 0444) == 0 && identity_at(pack).inode != identity_at(original_pack).inode &&
            identity_at(pack).size == identity_at(original_pack).size, "copy is another file of the same size");
    falls_back("copied pack and seal fall back");
    refreshes("copied pack is sealed after full verification");
    accepts("copied pack is then accepted");

    // Renamed: the same inode, but the change time moves.
    const auto renamed = copies / "renamed.asa.gguf";
    std::filesystem::rename(pack, renamed);
    std::filesystem::rename(seal, copies / "renamed.asa.seal");
    pack = renamed;
    seal = copies / "renamed.asa.seal";
    falls_back("renamed pack and seal fall back");
    refreshes("renamed pack is sealed after full verification");

    pack = original_pack;
    seal = original_seal;
    accepts("original pack and seal are unaffected by the copy");
}

void test_v3_authentication_sources(const std::filesystem::path& dir) {
    using namespace tokspur::planar_v3;
    std::filesystem::create_directory(dir);
    std::vector<SyntheticTensorSpec> tensors{
        {"token_embd.weight", 8, {256, 4}, make_valid_q8_rows(256, 4, 1)},
        {"per_layer_token_embd.weight", 0, {64, 2}, make_f32_bytes(128, 2)},
        {"output.weight", 8, {512, 256}, make_valid_q8_rows(512, 256, 3)},
        {"output_hc_norm.weight", 0, {20000}, make_f32_bytes(20000, 4)},
    };
    // Exercise every v3 authentication branch, including both halves of a
    // co-located gate/up unit and reads spanning the 32 KiB chunk boundary.
    for (uint32_t block = 0; block < 3; ++block) {
        const uint32_t type = block == 0 ? 12 : (block == 1 ? 13 : 8);
        const uint32_t down_type = block == 2 ? 8 : 7;
        const auto prefix = "blk." + std::to_string(block) + ".";
        for (const auto* part : {"ffn_gate_exps.weight", "ffn_up_exps.weight"}) {
            tensors.push_back({prefix + part, type, {256, 32, 4},
                               grouped_fixture::weights({type, 256, 32, 4, 1, 1, true})});
        }
        tensors.push_back({prefix + "ffn_down_exps.weight", down_type, {32, 256, 4},
                           grouped_fixture::weights({down_type, 32, 256, 4, 1, 1, true})});
    }
    const std::vector<SyntheticTensorSpec> mtp_tensors{
        {"blk.48.attn_q.weight", 8, {512, 128}, make_valid_q8_rows(512, 128, 5)},
    };
    const auto source_path = dir / "source.gguf", mtp_path = dir / "mtp.gguf";
    const auto pack_path = dir / "pack.asa.gguf";
    write_gguf_shard(source_path, "qwen35moe", tensors);
    write_gguf_shard(mtp_path, "qwen35moe", mtp_tensors);
    const auto source = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{source_path});
    const auto mtp = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{mtp_path});
    static_cast<void>(build_derived_pack_v3(source, mtp, pack_path, 2, 0, 32768));
    const auto adopt = [&](std::shared_ptr<const ModelFiles> origin, std::shared_ptr<const ModelFiles> sidecar,
                           uint32_t workers) {
        return adopt_derived_pack_v3(std::move(origin), std::move(sidecar), pack_path,
                                    2, 32768, workers, TrustMode::full, false);
    };
    const auto single = adopt(source, mtp, 1), parallel = adopt(source, mtp, 4);
    require(single.manifest_sha256 == parallel.manifest_sha256 && parallel.authentication_workers == 4,
            "v3 single and parallel authentication differ");
    const std::string_view expected_read_mode =
#if defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM) && TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM
        "bounded_nocache";
#else
        "mapped";
#endif
    require(single.authentication_source_read_mode == expected_read_mode &&
            parallel.authentication_source_read_mode == expected_read_mode,
            "v3 full authentication source-read mode mismatch");
    uint32_t layouts = 0;
    for (const auto& tensor : single.plan.tensors) layouts |= uint32_t{1} << uint32_t(tensor.kind);
    require(layouts == 0x3ffu, "v3 CPU authentication fixture must cover all ten tensor layouts");
    const MappedFile packed_file(pack_path);
    const auto packed_bytes = packed_file.bytes();
    std::cout << "v3_auth_source_read_mode=" << expected_read_mode
              << " manifest_sha256=" << single.manifest_sha256
              << " pack_sha256=" << hash(std::string_view(reinterpret_cast<const char*>(packed_bytes.data()), packed_bytes.size()))
              << '\n';
    for (size_t i = 0; i < 2; ++i) {
        require(std::ranges::equal(source->tensor_bytes(tensors[i].name), tensors[i].payload),
                "CPU embedding mapping changed during v3 authentication");
    }
    for (size_t i = 2; i < tensors.size(); ++i) {
        auto changed = tensors;
        changed[i].payload.back() ^= std::byte{1};
        const auto changed_path = dir / ("changed-" + std::to_string(i) + ".gguf");
        write_gguf_shard(changed_path, "qwen35moe", changed);
        const auto origin = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{changed_path});
        refuses([&] { static_cast<void>(adopt(origin, mtp, 4)); },
                "v3 full authentication accepted changed source tensor tail");
    }
    auto changed_mtp = mtp_tensors;
    changed_mtp[0].payload.back() ^= std::byte{1};
    const auto changed_mtp_path = dir / "changed-mtp.gguf";
    write_gguf_shard(changed_mtp_path, "qwen35moe", changed_mtp);
    const auto sidecar = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{changed_mtp_path});
    refuses([&] { static_cast<void>(adopt(source, sidecar, 4)); },
            "v3 full authentication accepted changed MTP source tail");
    {
        // tokspur.source.revision metadata propagates into target_revision and refuses mismatched sources.
        const std::string custom_rev = "34567a4712bc9766c4449e2e98e4468bfa24d915";
        const auto rev_source_path = dir / "rev-source.gguf";
        const auto rev_pack_path = dir / "rev-pack.asa.gguf";
        write_gguf_shard(rev_source_path, "qwen35moe", tensors, 0, 1, 0, custom_rev);
        const auto rev_source = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{rev_source_path});
        static_cast<void>(build_derived_pack_v3(rev_source, mtp, rev_pack_path, 2, 0, 32768));
        const auto rev_adopted = adopt_derived_pack_v3(rev_source, mtp, rev_pack_path, 2, 32768, 2, TrustMode::full, false);
        require(rev_adopted.plan.target_revision == custom_rev, "v3 pack recorded custom tokspur.source.revision");
        const auto wrong_rev_path = dir / "wrong-rev-source.gguf";
        write_gguf_shard(wrong_rev_path, "qwen35moe", tensors, 0, 1, 0, "0000000000000000000000000000000000000000");
        const auto wrong_rev_source = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{wrong_rev_path});
        refuses([&] {
            static_cast<void>(adopt_derived_pack_v3(wrong_rev_source, mtp, rev_pack_path, 2, 32768, 2, TrustMode::full, false));
        }, "v3 pack refused mismatched tokspur.source.revision");

        // 1-file CPU-base-only source enforces full 1,254-tensor census when reconstructing plan from pack.
        const std::vector<SyntheticTensorSpec> cpu_base_only{tensors[0], tensors[1]};
        const auto cpu_base_path = dir / "cpu-base-only.gguf";
        write_gguf_shard(cpu_base_path, "qwen4exp", cpu_base_only, 0, 1, 0, custom_rev);
        const auto cpu_base_source = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{cpu_base_path});
        refuses([&] {
            static_cast<void>(adopt_derived_pack_v3(cpu_base_source, mtp, rev_pack_path, 2, 32768, 2, TrustMode::full, false));
        }, "CPU-base-only v3 adoption enforces full 1,254-tensor census");
    }
    {
        // Local-seal mode covers full-census packs only. For any other v3 pack
        // a seal is neither read nor written, even one that would vouch.
        const DirFd pack_dir(dir);
        const auto pack_name = pack_path.filename().string();
        const auto seal_name = local_seal_name(pack_name).value();
        const auto partial = [&](std::shared_ptr<const ModelFiles> origin) {
            return adopt_derived_pack_v3(std::move(origin), mtp, pack_path, 2, 32768, 4, TrustMode::local_seal, false);
        };
        require(partial(source).local_seal == LocalSealUse::not_consulted && !std::filesystem::exists(dir / seal_name),
                "partial v3 pack is fully verified and gets no seal");
        publish_local_seal(pack_dir.fd, pack_name, {single.manifest_sha256, identity_at(pack_path), scope_both}, false);
        require(single.compared == scope_both && parallel.compared == scope_both,
                "v3 adoption with the target shards and the sidecar compares both");
        require(read_local_seal(pack_dir.fd, pack_name, single.manifest_sha256, identity_at(pack_path), scope_both).vouches &&
                single.pack_identity == identity_at(pack_path), "vouching seal placed beside a partial v3 pack");
        auto changed = tensors;
        changed[2].payload.back() ^= std::byte{1};
        write_gguf_shard(dir / "seal-changed.gguf", "qwen35moe", changed);
        const auto origin = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{dir / "seal-changed.gguf"});
        refuses([&] { static_cast<void>(partial(origin)); }, "partial v3 pack ignores a vouching seal");
        const auto adopted = partial(source);
        require(adopted.local_seal == LocalSealUse::not_consulted && adopted.authentication_workers == 4 &&
                std::filesystem::remove(dir / seal_name), "partial v3 pack never consults a seal");
    }
#if defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM) && TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM
    // The bounded descriptor must bind the same file as the retained mapping.
    std::filesystem::rename(source_path, dir / "original-source.gguf");
    write_gguf_shard(source_path, "qwen35moe", tensors);
    refuses([&] { static_cast<void>(adopt(source, mtp, 4)); },
            "v3 nocache authentication accepted a replaced source inode");
#endif
}

void test_v3_publication(const std::filesystem::path& dir, std::shared_ptr<const ModelFiles> source,
                         std::shared_ptr<const ModelFiles> mtp) {
    using namespace tokspur::planar_v3;
    std::filesystem::create_directory(dir);
    const auto output = dir / "block.asa.gguf";
    const auto seal = dir / "block.asa.seal";
    const auto full = plan_derived_pack_v3(*source, mtp.get());
    refuses([&] { static_cast<void>(build_or_verify_pack_v3(full, source, mtp, output, 2, 0, 32768)); },
            "partial target cannot publish a full v3 recipe");
    const auto block = plan_block_pack_v3(*source, mtp.get(), 0);
    for (const auto* reserved : {"tokspur-pack-v2.asa.gguf", "TokSpur-Pack-V2.Gguf", "wrong.seal"}) {
        refuses([&] { static_cast<void>(build_or_verify_pack_v3(block, source, mtp, dir / reserved, 2, 0, 32768)); },
                "reserved pack or seal destination refused");
    }
    const auto budget = measure_pack_v3_budget(block, dir, {}, 2, 0, 32768);
    require(budget.planned_pack_v3_file_bytes == block.file_bytes && budget.streaming_ring_buffer_bytes >= 65536,
            "pack v3 budget accounts for configured Q8 write and readback buffers");
    const auto publish = [&](uint64_t reserve = 0) {
        return build_or_verify_pack_v3(block, source, mtp, output, 2, reserve, 32768);
    };
    refuses([&] {
        static_cast<void>(build_or_verify_pack_v3(block, source, mtp, output, 2, 0, 32768,
                                                  [] { throw FormatError("interrupted test build"); }));
    }, "interruption refused");
    require(count_entries(dir) == 0, "interrupted write left no partial pack or seal");

    const int lock = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    require(lock >= 0 && ::flock(lock, LOCK_EX | LOCK_NB) == 0, "acquire competing output directory lock");
    refuses([&] { static_cast<void>(publish()); }, "competing v3 publisher refused");
    ::close(lock);
    require(count_entries(dir) == 0, "competing writer left no output");

    write_file_bytes(seal, Bytes(65, std::byte{'0'}), 0444);
    refuses([&] { static_cast<void>(publish()); }, "orphan v3 seal refused");
    require(!std::filesystem::exists(output) && std::filesystem::remove(seal), "orphan seal preserved");
    require(::symlink("missing.gguf", output.c_str()) == 0, "create dangling pack symlink");
    refuses([&] { static_cast<void>(publish()); }, "dangling v3 output symlink refused");
    require(std::filesystem::is_symlink(output) && std::filesystem::remove(output), "output symlink preserved");

    write_file_bytes(output, Bytes(64, std::byte{'x'}), 0444);
    refuses([&] { static_cast<void>(publish()); }, "incompatible existing pack refused");
    require(std::filesystem::file_size(output) == 64 && !std::filesystem::exists(seal) &&
            std::filesystem::remove(output), "incompatible pack never replaced or sealed");
    const auto built = publish();
    require(!built.existed && built.verified.manifest_sha256.size() == 64, "built and fully authenticated new v3 pack");
    // The seal names the published file by the identity it has after the read-only chmod and the rename.
    const DirFd pack_dir(dir);
    const auto& manifest = built.verified.manifest_sha256;
    const auto sealed = [&](SealScope needed = scope_both) {
        return read_local_seal(pack_dir.fd, "block.asa.gguf", manifest, identity_at(output), needed,
                               SealCertification::dm_fp16_v1);
    };
    require(built.verified.local_seal == LocalSealUse::not_consulted && built.verified.pack_identity == identity_at(output) &&
            built.verified.compared == scope_both &&
            read_text(seal) == encode_local_seal({manifest, identity_at(output), scope_both, SealCertification::dm_fp16_v1}) &&
            sealed().vouches && sealed(scope_target).vouches && sealed(scope_mtp).vouches && sealed(scope_none).vouches,
            "published v3 seal records the verified pack file, both sources, and dm-fp16-v1 certification");
    struct stat pack_st{}, seal_st{};
    require(::stat(output.c_str(), &pack_st) == 0 && ::stat(seal.c_str(), &seal_st) == 0 &&
            (pack_st.st_mode & 0777) == 0444 && (seal_st.st_mode & 0777) == 0444, "v3 pack and seal immutable");
    const auto existing = publish(std::numeric_limits<uint64_t>::max());
    struct stat same{};
    require(existing.existed && existing.verified.manifest_sha256 == built.verified.manifest_sha256 &&
            ::stat(seal.c_str(), &same) == 0 && same.st_ino == seal_st.st_ino, "existing v3 seal was reused");
    require(std::filesystem::remove(seal), "simulate interruption after pack publication");
    const auto repaired = publish();
    require(repaired.existed && repaired.verified.manifest_sha256 == built.verified.manifest_sha256 &&
            ::stat(output.c_str(), &same) == 0 && same.st_ino == pack_st.st_ino, "missing v3 seal repaired in place");

    // A sound seal for this manifest that is not current (including uncertified seal-v2) is replaced after full verification.
    for (const auto& stale : {manifest + "\n", manifest,
                              encode_local_seal({manifest, {}, scope_both, SealCertification::dm_fp16_v1}),
                              encode_local_seal({manifest, identity_at(output), scope_both, SealCertification::none}),
                              // The right file, compared with fewer sources than the publisher just compared.
                              encode_local_seal({manifest, identity_at(output), scope_target, SealCertification::dm_fp16_v1}),
                              encode_local_seal({manifest, identity_at(output), scope_mtp, SealCertification::dm_fp16_v1}),
                              encode_local_seal({manifest, identity_at(output), scope_none, SealCertification::dm_fp16_v1})}) {
        put_text(seal, stale);
        require(::stat(seal.c_str(), &seal_st) == 0 && !sealed().vouches && sealed().names_manifest, "stale seal for this manifest");
        const auto resealed = publish();
        require(resealed.existed && sealed().vouches && ::stat(seal.c_str(), &same) == 0 && same.st_ino != seal_st.st_ino &&
                (same.st_mode & 0777) == 0444 && count_entries(dir) == 2 && resealed.verified.compared == scope_both &&
                read_text(seal) == encode_local_seal({manifest, identity_at(output), scope_both, SealCertification::dm_fp16_v1}),
                "stale or uncertified v2 seal for this manifest is upgraded to certified v3 seal");
    }
    // Anything else at the seal name is refused and left exactly as it was.
    const std::string other_manifest(64, '0');
    for (const auto& foreign : {encode_local_seal({other_manifest, identity_at(output), scope_both, SealCertification::dm_fp16_v1}),
                                std::string("garbage\n")}) {
        put_text(seal, foreign);
        require(::stat(seal.c_str(), &seal_st) == 0, "stat foreign seal");
        refuses([&] { static_cast<void>(publish()); }, "seal for another manifest refused");
        require(::stat(seal.c_str(), &same) == 0 && same.st_ino == seal_st.st_ino && read_text(seal) == foreign,
                "seal for another manifest never replaced");
    }
    put_text(seal, encode_local_seal({manifest, identity_at(output), scope_both, SealCertification::dm_fp16_v1}), 0644);
    require(::stat(seal.c_str(), &seal_st) == 0 && (seal_st.st_mode & S_IWUSR), "fixture seal must be owner-writable");
    refuses([&] { static_cast<void>(publish()); }, "writable v3 seal refused");
    require(::stat(seal.c_str(), &same) == 0 && same.st_ino == seal_st.st_ino && same.st_mode == seal_st.st_mode,
            "writable seal never replaced");

    require(std::filesystem::remove(seal), "remove valid seal for mismatch test");
    write_file_bytes(seal, Bytes(64, std::byte{'0'}), 0444);
    require(::stat(seal.c_str(), &seal_st) == 0, "stat wrong seal");
    refuses([&] { static_cast<void>(publish()); }, "mismatched v3 seal refused");
    require(::stat(seal.c_str(), &same) == 0 && same.st_ino == seal_st.st_ino, "mismatched seal never replaced");
    require(std::filesystem::remove(seal), "remove wrong seal");
    require(::symlink("missing.seal", seal.c_str()) == 0, "create dangling seal symlink");
    refuses([&] { static_cast<void>(publish()); }, "dangling v3 seal symlink refused");
    require(std::filesystem::is_symlink(seal) && std::filesystem::remove(seal), "seal symlink preserved");
    static_cast<void>(publish());

    // A matching manifest seal must not bypass full source verification on retry.
    require(::chmod(output.c_str(), 0600) == 0, "unseal fixture");
    const int fd = ::open(output.c_str(), O_RDWR | O_CLOEXEC);
    std::byte byte{};
    const auto offset = static_cast<off_t>(block.data_offset);
    require(fd >= 0 && ::pread(fd, &byte, 1, offset) == 1, "read v3 test byte");
    byte ^= std::byte{1};
    require(::pwrite(fd, &byte, 1, offset) == 1 && ::fchmod(fd, 0444) == 0, "alter v3 test byte");
    ::close(fd);
    refuses([&] { static_cast<void>(publish()); }, "tampered v3 payload refused even with a matching seal");
    require(std::filesystem::remove(seal), "remove seal on corrupt fixture");
    refuses([&] { static_cast<void>(publish()); }, "tampered v3 payload cannot be resealed");
    require(!std::filesystem::exists(seal) && ::stat(output.c_str(), &same) == 0 &&
            same.st_ino == pack_st.st_ino, "corrupt pack preserved and no new seal issued");
}

void test_v3_static_dm_and_fp16_certification(const std::filesystem::path& dir) {
    using namespace tokspur::planar_v3;
    std::filesystem::create_directory(dir);

    // 1. Direct pack_expert_planar_v3 & certify_expert_planar_v3 reject NaN/Inf dm and FP16-staging overflows.
    for (const uint32_t st : {12u, 13u, 7u, 8u}) {
        const uint32_t cols = (st == 12u || st == 13u) ? 256u : 32u;
        const uint32_t rows = (st == 8u) ? 8u : 4u;
        const auto spec = make_expert_spec(st, cols, rows);
        const Bytes ok_gguf(spec.source_expert_bytes, std::byte{0});
        Bytes ok_planar(spec.planar_expert_bytes, std::byte{0});
        pack_expert_planar_v3(spec, ok_gguf, ok_planar);
        certify_expert_planar_v3(spec, ok_planar);

        for (const uint16_t bad_dm : {uint16_t(0x7e00u), uint16_t(0x7c00u), uint16_t(0xfc00u)}) {
            Bytes bad_gguf = ok_gguf;
            std::memcpy(bad_gguf.data(), &bad_dm, 2);
            Bytes scratch(spec.planar_expert_bytes, std::byte{0});
            refuses([&] { pack_expert_planar_v3(spec, bad_gguf, scratch); },
                    "pack_expert_planar_v3 must reject nonfinite dm");
            Bytes bad_planar = ok_planar;
            std::memcpy(bad_planar.data() + spec.dm_offset, &bad_dm, 2);
            refuses([&] { certify_expert_planar_v3(spec, bad_planar); },
                    "certify_expert_planar_v3 must reject nonfinite dm");
        }

        // Large finite scale 65504 (0x7bff) with q=0 is safe; with large q it overflows FP16 staging!
        const uint16_t large_h = 0x7bffu;
        Bytes zero_q_planar = ok_planar;
        std::memcpy(zero_q_planar.data() + spec.dm_offset, &large_h, 2);
        certify_expert_planar_v3(spec, zero_q_planar);

        Bytes overflow_planar = zero_q_planar;
        if (st == 12u || st == 13u) {
            overflow_planar[size_t(spec.sm_offset)] = std::byte{63};
            std::fill_n(overflow_planar.begin() + std::ptrdiff_t(spec.qs_offset), 128u, std::byte{0x7f});
        } else if (st == 7u) {
            std::fill_n(overflow_planar.begin() + std::ptrdiff_t(spec.qs_offset), 16u, std::byte{0x7f});
        } else {
            std::fill_n(overflow_planar.begin() + std::ptrdiff_t(spec.qs_offset), 32u, std::byte{0x7f});
        }
        refuses([&] { certify_expert_planar_v3(spec, overflow_planar); },
                "certify_expert_planar_v3 must reject FP16 staging overflow");

        if (st == 12u || st == 13u) {
            Bytes bad_sm = ok_planar;
            bad_sm[size_t(spec.sm_offset)] = std::byte{64};
            refuses([&] { certify_expert_planar_v3(spec, bad_sm); },
                    "certify_expert_planar_v3 must reject sm > 63");
        }
    }

    // 2. build_derived_pack_v3 refuses GGUF source containing FP16-staging overflow.
    {
        Bytes bad_q51 = make_valid_q51_rows(640, 8, 9);
        const uint16_t large_h = 0x7bffu;
        std::memcpy(bad_q51.data(), &large_h, 2);
        std::fill_n(bad_q51.begin() + 8, 16, std::byte{0x7f});
        const std::vector<SyntheticTensorSpec> bad_tensors{
            {"blk.0.ffn_down_exps.weight", 7, {640, 4, 2}, std::move(bad_q51)},
        };
        const auto bad_src_path = dir / "bad-fp16-src.gguf";
        const auto bad_mtp_path = dir / "bad-fp16-mtp.gguf";
        write_gguf_shard(bad_src_path, "qwen35moe", bad_tensors);
        write_gguf_shard(bad_mtp_path, "qwen35moe", {{"blk.48.attn_q.weight", 8, {256, 8}, make_valid_q8_rows(256, 8, 1)}});
        const auto bad_src = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{bad_src_path});
        const auto bad_mtp = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{bad_mtp_path});
        refuses([&] {
            static_cast<void>(build_derived_pack_v3(bad_src, bad_mtp, dir / "bad-fp16-out.asa.gguf", 2, 0, 32768));
        }, "build_derived_pack_v3 must reject FP16-staging overflow in source");
    }

    // 3. adopt_derived_pack_v3 certifies pack_only tensors (when mtp_sidecar == nullptr) & rejects corrupt blk.48.* dm.
    {
        const auto ok_src_path = dir / "src-ok.gguf";
        const auto ok_mtp_path = dir / "mtp-ok.gguf";
        const auto pk_path = dir / "pack-ok.asa.gguf";
        const std::vector<SyntheticTensorSpec> src_t{
            {"blk.0.ffn_down_exps.weight", 7, {640, 4, 2}, make_valid_q51_rows(640, 4 * 2, 3)},
        };
        const std::vector<SyntheticTensorSpec> mtp_t{
            {"blk.48.ffn_down_exps.weight", 7, {640, 4, 2}, make_valid_q51_rows(640, 4 * 2, 4)},
        };
        write_gguf_shard(ok_src_path, "qwen35moe", src_t);
        write_gguf_shard(ok_mtp_path, "qwen35moe", mtp_t);
        const auto src_m = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{ok_src_path});
        const auto mtp_m = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{ok_mtp_path});
        const auto plan = build_derived_pack_v3(src_m, mtp_m, pk_path, 2, 0, 32768);
        const auto adopted_pack_only = adopt_pack_v3_with_plan(plan, src_m, nullptr, pk_path, 2, 32768, 1,
                                                               TrustMode::full, false);
        require(is_certified_pack_model(adopted_pack_only.model),
                "adopted Pack v3 model must be registered as certified");
    }

    // 4. Legacy tokspur-local-seal-v2 cannot vouch for Pack v3.
    {
        const std::string manifest(64, 'a');
        const PackFileIdentity id{111, 222, 1000, 0, 1000, 0, 1000, 0};
        const auto v2_seal = encode_local_seal({manifest, id, scope_both, SealCertification::none});
        const auto v3_seal = encode_local_seal({manifest, id, scope_both, SealCertification::dm_fp16_v1});
        require(local_seal_vouches(v2_seal, manifest, id, scope_both, SealCertification::none),
                "v2 seal vouches when no extra certification required (Pack v2)");
        require(!local_seal_vouches(v2_seal, manifest, id, scope_both, SealCertification::dm_fp16_v1),
                "uncertified v2 seal must NOT vouch for Pack v3 dm_fp16_v1");
        require(local_seal_vouches(v3_seal, manifest, id, scope_both, SealCertification::dm_fp16_v1),
                "certified v3 seal vouches for Pack v3 dm_fp16_v1");
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && (std::string_view(argv[1]) == "--verify-pack" || std::string_view(argv[1]) == "--adopt")) {
        return run_cli_verify(argc, argv);
    }

    try {
        TempDir root;
        const auto source_dir = root.path / "source";
        const auto out_dir = root.path / "out";
        std::filesystem::create_directories(source_dir);
        std::filesystem::create_directories(out_dir);

        // Multi-chunk Q8_0 tensor (> 64 KiB so 32 KiB chunks split across 3 iterations).
        const std::vector<SyntheticTensorSpec> shard0 = {
            {"token_embd.weight", 8, {256, 4}, make_valid_q8_rows(256, 4, 1)},
            {"per_layer_token_embd.weight", 0, {64, 2}, make_f32_bytes(128, 2)},
            {"blk.0.attn_q.weight", 8, {256, 256}, make_valid_q8_rows(256, 256, 3)},
            {"blk.0.ffn_down_exps.weight", 7, {640, 4, 2}, make_valid_q51_rows(640, 8, 4)},
        };
        const std::vector<SyntheticTensorSpec> shard1 = {
            {"blk.1.ple_key.weight", 8, {256, 4}, make_valid_q8_rows(256, 4, 5)},
            {"output_hc_norm.weight", 0, {128}, make_f32_bytes(128, 6)},
        };
        const std::vector<SyntheticTensorSpec> mtp_specs = {
            {"blk.48.attn_q.weight", 8, {256, 8}, make_valid_q8_rows(256, 8, 7)},
            {"blk.48.ffn_gate_inp.weight", 0, {64, 2}, make_f32_bytes(128, 8)},
        };

        const auto shard0_path = source_dir / "shard-00001-of-00002.gguf";
        const auto shard1_path = source_dir / "shard-00002-of-00002.gguf";
        const auto mtp_path = source_dir / "mtp.gguf";
        write_gguf_shard(shard0_path, "qwen4exp", shard0, 0, 2, uint32_t(shard0.size() + shard1.size()));
        write_gguf_shard(shard1_path, "qwen4exp", shard1, 1, 2, uint32_t(shard0.size() + shard1.size()));
        write_gguf_shard(mtp_path, "qwen4exp", mtp_specs);

        auto source = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{shard0_path, shard1_path});
        auto mtp = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{mtp_path});

        test_local_seal_format();
        test_local_seal_files(root.path / "seal-files");
        test_local_seal_adoption(root.path / "seal-adoption");
        test_v3_publication(root.path / "v3", source, mtp);
        test_v3_static_dm_and_fp16_certification(root.path / "v3-cert");
        test_v3_authentication_sources(root.path / "v3-auth");

        // 1. Planning checks
        const auto planned = plan_derived_pack(*source, mtp.get());
        require(planned.tensors.size() == 6, "CPU embeddings excluded and GPU + MTP tensors retained");
        for (const auto& t : planned.tensors) {
            require(!is_cpu_pinned_embedding(t.name), "CPU-pinned tensor leaked into pack plan");
            require(t.offset % pack_alignment == 0, "tensor offset aligned to 16 KiB");
            if (t.name == "blk.0.attn_q.weight" || t.name == "blk.48.attn_q.weight") {
                require(t.layout == "q8_lanes_v1", "eligible 2D Q8_0 tensor assigned q8_lanes_v1");
            } else if (t.name == "blk.0.ffn_down_exps.weight") {
                require(t.layout == "q51_lanes_v1", "eligible 3D Q5_1 down_exps assigned q51_lanes_v1");
            } else {
                require(t.layout == "plain", "non-eligible tensor kept plain");
            }
        }
        refuses([&] {
            static_cast<void>(plan_derived_pack(*source, mtp.get(), planned.file_bytes - 1));
        }, "byte_budget below planned file size refused");

        // 2. Reserve refusal (preflight, overflow, and deterministic mid-stream chunk drop)
        struct statvfs vfs{};
        require(::statvfs(out_dir.c_str(), &vfs) == 0, "statvfs on out_dir");
        const uint64_t real_avail = uint64_t(vfs.f_bavail) * uint64_t(vfs.f_frsize);
        const auto refused_out = out_dir / "refused-pack.gguf";
        // Reserve 1 TiB beyond the measured free space, so space that other jobs on a shared runner free before the build's own fstatvfs read is too small to admit the build.
        refuses([&] {
            static_cast<void>(build_derived_pack(source, mtp, refused_out, real_avail + (1ULL << 40), 32'768));
        }, "preflight reserve deficit refused");
        require(count_entries(out_dir) == 0, "no temporary files left after preflight reserve refusal");

        refuses([&] {
            static_cast<void>(build_derived_pack(source, mtp, refused_out, std::numeric_limits<uint64_t>::max(), 32'768));
        }, "overflowing reserve requirement refused");
        require(count_entries(out_dir) == 0, "no temporary files left after overflow reserve refusal");

        size_t space_calls = 0;
        const uint64_t test_reserve = 50ULL * 1024 * 1024 * 1024;
        refuses([&] {
            static_cast<void>(build_derived_pack_with_space_query(
                source,
                mtp,
                refused_out,
                test_reserve,
                32'768,
                [&](int) -> uint64_t {
                    ++space_calls;
                    // Allow preflight (call 1) and chunk 0 of tensor 0 (call 2), then drop below reserve on chunk 1 (call 3).
                    if (space_calls <= 2) {
                        return planned.file_bytes + test_reserve + (1ULL << 20);
                    }
                    return test_reserve + 1024;
                }
            ));
        }, "mid-stream chunk reserve drop refused before consuming reserve");
        require(space_calls == 3, "space query ran on preflight and every chunk write");
        require(count_entries(out_dir) == 0, "RAII temporary unlinked on mid-stream reserve refusal");

        // 3. Build and exclusive publication checks
        const auto pack_path = out_dir / "tokspur-pack-v2.asa.gguf";
        const auto built = build_derived_pack(source, mtp, pack_path, 1ULL << 20, 32'768);
        require(built.file_bytes == planned.file_bytes, "built file_bytes matches plan");
        require(count_entries(out_dir) == 1, "only sealed pack present in out_dir");

        struct stat st{};
        require(::lstat(pack_path.c_str(), &st) == 0, "lstat on built pack");
        require(S_ISREG(st.st_mode) && (st.st_mode & 0777) == 0444, "built pack sealed 0444");
        require(uint64_t(st.st_size) == planned.file_bytes, "built pack exact byte length without ftruncate");

        // Refuses to overwrite existing regular file
        refuses([&] {
            static_cast<void>(build_derived_pack(source, mtp, pack_path, 1ULL << 20, 32'768));
        }, "existing pack file refused without overwrite");
        require(count_entries(out_dir) == 1, "no leaked temporary after existing-file refusal");

        // Refuses to overwrite existing symlink (including dangling symlink)
        const auto symlink_out = out_dir / "dangling-link.gguf";
        require(::symlink("nonexistent-target", symlink_out.c_str()) == 0, "create dangling symlink");
        refuses([&] {
            static_cast<void>(build_derived_pack(source, mtp, symlink_out, 1ULL << 20, 32'768));
        }, "existing dangling symlink refused");
        require(::unlink(symlink_out.c_str()) == 0, "remove dangling symlink");

        // Refuses symlink parent directory component via O_NOFOLLOW traversal
        const auto sym_dir = root.path / "sym_out";
        require(::symlink(out_dir.c_str(), sym_dir.c_str()) == 0, "create directory symlink");
        refuses([&] {
            static_cast<void>(build_derived_pack(source, mtp, sym_dir / "via-symlink.gguf", 1ULL << 20, 32'768));
        }, "symlink directory component refused");
        require(::unlink(sym_dir.c_str()) == 0, "remove directory symlink");

        // 4. Adoption and tampering checks
        refuses([&] {
            static_cast<void>(adopt_derived_pack(source, mtp, pack_path, 32'768, 1));
        }, "default adopt_derived_pack enforces full 1,254-tensor census on partial fixture");
        refuses([&] {
            static_cast<void>(adopt_derived_pack(source, mtp, pack_path, 32'768, 1, TrustMode::local_seal, false));
        }, "local-seal always enforces full 1,254-tensor census");
        const auto adopted1 = adopt_derived_pack(source, mtp, pack_path, 32'768, 1, TrustMode::full, false);
        const auto adopted4 = adopt_derived_pack(source, mtp, pack_path, 32'768, 4, TrustMode::full, false);
        require(adopted1.manifest_sha256 == adopted4.manifest_sha256, "single and multi-worker adoption match");
        require(adopted1.model->architecture() == pack_architecture, "adopted pack architecture");
        auto altered_source = shard0;
        altered_source[2].payload.back() ^= std::byte(0x01);
        const auto altered_source_path = source_dir / "altered-source.gguf";
        write_gguf_shard(altered_source_path, "qwen4exp", altered_source, 0, 2, uint32_t(shard0.size() + shard1.size()));
        const auto changed_source = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{altered_source_path, shard1_path});
        refuses([&] {
            static_cast<void>(adopt_derived_pack(changed_source, mtp, pack_path, 32'768, 4, TrustMode::full, false));
        }, "corrupted original tail byte refused across authentication chunks");
        auto altered_mtp = mtp_specs;
        altered_mtp.back().payload.back() ^= std::byte(0x01);
        const auto altered_mtp_path = source_dir / "altered-mtp.gguf";
        write_gguf_shard(altered_mtp_path, "qwen4exp", altered_mtp);
        const auto changed_mtp = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{altered_mtp_path});
        refuses([&] {
            static_cast<void>(adopt_derived_pack(source, changed_mtp, pack_path, 32'768, 4, TrustMode::full, false));
        }, "corrupted original MTP bytes refused during authentication");
        refuses([&] {
            static_cast<void>(plan_derived_pack(*adopted1.model, nullptr));
        }, "repacking tokspur_pack refused");

        // Writable pack refused
        const auto bad_pack = out_dir / "tampered.gguf";
        std::filesystem::copy_file(pack_path, bad_pack);
        require(::chmod(bad_pack.c_str(), 0644) == 0, "make copy writable");
        refuses([&] {
            static_cast<void>(adopt_derived_pack(source, mtp, bad_pack, 32'768, 1, TrustMode::full, false));
        }, "writable pack refused by adopt_derived_pack");

        // Flip 1 byte in first tensor payload -> refused even when re-sealed 0444
        {
            const int fd = ::open(bad_pack.c_str(), O_RDWR | O_CLOEXEC);
            require(fd >= 0, "open tampered pack");
            std::byte b{};
            require(::pread(fd, &b, 1, off_t(built.data_offset)) == 1, "read payload byte");
            b ^= std::byte(0x01);
            require(::pwrite(fd, &b, 1, off_t(built.data_offset)) == 1, "flip payload byte");
            require(::fchmod(fd, 0444) == 0, "re-seal tampered pack");
            ::close(fd);
        }
        refuses([&] {
            static_cast<void>(adopt_derived_pack(source, mtp, bad_pack, 32'768, 2, TrustMode::full, false));
        }, "corrupted payload byte refused by adopt_derived_pack");

        std::cout << "PASS derived pack checks=" << checks << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
