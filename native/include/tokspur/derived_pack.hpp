#pragma once
// Private, Darwin-only lossless derived weight pack (tokspur-pack-v2).
// Implements 16 KiB page-aligned Metal mapping, byte-exact inverse reconstruction proof,
// 64-bit uint64_t addressing, SHA-256 integrity, and disk reserve verification.

#include "tokspur/gguf.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tokspur::derived_pack {

inline constexpr uint64_t pack_alignment = 16384; // 16 KiB page boundary for Metal newBufferWithBytesNoCopy
inline constexpr uint32_t pack_version = 2;
inline constexpr const char* pack_recipe = "tokspur-pack-v2";
inline constexpr const char* pack_architecture = "tokspur_pack";
inline constexpr const char* source_architecture = "qwen4exp";
inline constexpr const char* target_revision = "34567a4712bc9766c4449e2e98e4468bfa24d915";
inline constexpr const char* mtp_sidecar_sha256 = "5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6";
inline constexpr size_t expected_gpu_tensors = 1254;

inline constexpr uint64_t default_reserve_bytes = 50ULL * 1024 * 1024 * 1024; // 50 GiB safety reserve
inline constexpr uint64_t default_chunk_bytes = 4'194'304; // 4 MiB streaming chunk
inline constexpr uint32_t max_authentication_workers = 16;
inline constexpr uint64_t max_pack_bytes = 128'000'000'000ULL; // 128 GB safety ceiling

struct PackTensor {
    std::string name;
    uint32_t type = 0;
    std::string layout = "plain";
    std::vector<uint64_t> dimensions;
    uint64_t bytes = 0;
    uint64_t offset = 0; // Aligned relative to data_offset
    std::string source_sha256;
    std::string packed_sha256;
};

struct PackPlan {
    std::vector<PackTensor> tensors;
    uint64_t payload_bytes = 0;
    uint64_t file_bytes = 0;
    uint64_t data_offset = 0;
    std::string recipe = pack_recipe;
    uint32_t version = pack_version;
    uint32_t alignment = pack_alignment;
};

using Bytes = std::vector<std::byte>;

// Compute SHA-256 hex digest of text
[[nodiscard]] std::string hash(std::string_view text);

// Build GGUF header bytes for pack plan
[[nodiscard]] Bytes build_header(const PackPlan& plan);

// The identity of one pack file, read with fstat from an open descriptor.
// The device number is left out: it is not stable across mounts.
struct PackFileIdentity {
    uint64_t inode = 0;
    uint64_t size = 0;
    int64_t mtime_seconds = 0;
    uint32_t mtime_nanoseconds = 0;
    int64_t ctime_seconds = 0;
    uint32_t ctime_nanoseconds = 0;
    int64_t birthtime_seconds = 0;
    uint32_t birthtime_nanoseconds = 0;
    bool operator==(const PackFileIdentity&) const = default;
};

// Which source files a full verification compared the pack with. target: the
// tensors the target shards hold were compared with them. mtp: the blk.48.*
// tensors were compared with the MTP sidecar. A tensor whose source was not
// given to the call is only checked against the pack's own manifest digests.
struct SealScope {
    bool target = false;
    bool mtp = false;
    bool operator==(const SealScope&) const = default;
    // True when a verification with this scope leaves nothing for a start that would compare `needed`.
    [[nodiscard]] constexpr bool covers(const SealScope& needed) const {
        return (target || !needed.target) && (mtp || !needed.mtp);
    }
};
[[nodiscard]] constexpr SealScope operator|(const SealScope& a, const SealScope& b) {
    return {a.target || b.target, a.mtp || b.mtp};
}

// What a local seal records: the manifest of a pack that passed full
// verification, the identity that pack file had when it passed, and which
// source files it was compared with.
struct LocalSeal {
    std::string manifest_sha256;
    PackFileIdentity pack;
    SealScope compared;
    bool operator==(const LocalSeal&) const = default;
};

// First line of a current seal. The earlier form was the bare manifest hash.
inline constexpr std::string_view local_seal_format = "tokspur-local-seal-v2";
inline constexpr size_t max_local_seal_bytes = 4096;

// The one accepted encoding of a seal, eight lines each ending in "\n":
//   tokspur-local-seal-v2
//   manifest_sha256=<64 lowercase hex digits>
//   compared=<target+mtp | target | mtp | none>
//   inode=<decimal>
//   size=<decimal>
//   mtime=<seconds>.<nine digit nanoseconds>
//   ctime=<seconds>.<nine digit nanoseconds>
//   birthtime=<seconds>.<nine digit nanoseconds>
[[nodiscard]] std::string encode_local_seal(const LocalSeal& seal);

enum class LocalSealForm {
    current,   // exactly an encode_local_seal result
    legacy,    // the earlier form: the manifest hash alone, which binds no file
    malformed, // anything else
};
struct ParsedLocalSeal {
    LocalSealForm form = LocalSealForm::malformed;
    LocalSeal seal; // manifest_sha256 is also set for the legacy form
};
[[nodiscard]] ParsedLocalSeal parse_local_seal(std::string_view text);

// True only for a current-format seal whose manifest and every identity field
// equal the ones given and whose scope covers `needed`, the sources this start
// would compare the pack with. This is the whole acceptance rule of local-seal mode.
[[nodiscard]] bool local_seal_vouches(std::string_view seal_text, std::string_view manifest_sha256,
                                      const PackFileIdentity& pack, SealScope needed);

// fstat on an open descriptor. Refuses anything but a regular file.
[[nodiscard]] PackFileIdentity pack_file_identity(int fd);

// <stem>.seal, the name a pack's seal has beside it. A pack whose own name is
// that name (it ends in .seal) has no seal name: it can never be sealed, so a
// seal can never be written over it.
[[nodiscard]] std::optional<std::string> local_seal_name(const std::string& pack_name);

struct LocalSealReading {
    enum class Entry { absent, regular, other };
    // What the seal name refers to, without following a link. regular: a
    // regular file of at most max_local_seal_bytes. other: anything else there,
    // or a pack that has no seal name; never used and never replaced.
    Entry entry = Entry::absent;
    bool trusted_file = false;           // regular, not writable, single link, owned by this user or root, 1..4096 bytes, read in full
    LocalSealForm form = LocalSealForm::malformed;
    bool names_manifest = false;         // current or legacy form carrying the given manifest hash
    SealScope compared;                  // of a current-form seal
    bool names_file = false;             // trusted_file, current form, manifest and every identity field equal; scope aside
    bool vouches = false;                // names_file and compared.covers(needed)
};
// Reads the seal beside the pack called pack_name and compares it. Never
// throws for a seal it cannot use: every such case is reported as a seal that
// does not vouch.
[[nodiscard]] LocalSealReading read_local_seal(int directory_fd, const std::string& pack_name,
                                               std::string_view manifest_sha256, const PackFileIdentity& pack,
                                               SealScope needed);

// Writes the seal for the pack called pack_name under a temporary name, makes
// it read-only, syncs it and renames it to the pack's seal name, so no reader
// sees a partial seal. With replace false the rename is exclusive. Callers
// must have just fully verified the file whose identity the seal carries,
// against the sources its scope names. Throws FormatError on failure, and
// before creating anything when the pack has no seal name.
void publish_local_seal(int directory_fd, const std::string& pack_name, const LocalSeal& seal, bool replace);

// For adoption in local-seal mode, after a full verification of the pack open
// as pack_fd has passed against the sources in `verified`: leaves a current
// seal beside pack_name. A seal that already names this manifest and exactly
// this file keeps its scope and gains `verified`. Any other regular seal is
// replaced by one carrying `verified` alone. Writes nothing and returns false
// when the pack has no seal name, when the file changed while it was verified
// (before differs from its identity now), when pack_name no longer names it,
// when the seal name is not a regular file or absent, when another writer
// holds the directory, or when the directory cannot be written.
[[nodiscard]] bool refresh_local_seal(int directory_fd, const std::string& pack_name, int pack_fd,
                                      std::string_view manifest_sha256, const PackFileIdentity& before,
                                      SealScope verified) noexcept;

// How adoption used the local seal.
enum class LocalSealUse {
    not_consulted, // full trust mode, or a pack kind the mode does not cover
    accepted,      // the seal vouched for the pack: no payload byte was read
    refreshed,     // no usable seal: fully verified, and a current seal is now in place
    not_refreshed, // no usable seal: fully verified, but no current seal could be left behind
};

struct AdoptedPack {
    std::shared_ptr<const ModelFiles> model;
    PackPlan plan;
    std::string manifest_sha256;
    uint32_t authentication_workers = 1;
    LocalSealUse local_seal = LocalSealUse::not_consulted;
    PackFileIdentity pack_identity; // of the descriptor that was adopted, after any verification
    SealScope compared;             // what this call fully verified the pack against; empty when the seal was accepted
};

// Check if a tensor name is a CPU-pinned embedding table (retained on host CPU)
[[nodiscard]] inline constexpr bool is_cpu_pinned_embedding(std::string_view name) noexcept {
    return name == "token_embd.weight" || name == "per_layer_token_embd.weight" || name == "per_layer_token_embd";
}

// Check if a tensor name is eligible for GPU packing (excludes CPU embedding tables)
[[nodiscard]] bool is_gpu_pack_tensor(std::string_view name);

// Verify census-driven GPU prototype checks across real GDN, layer-3 QSA, layer-1 ple_*, global output HC, output.weight, and blk.48 MTP
[[nodiscard]] bool verify_gpu_pack_census(const PackPlan& plan);

// Plan derived pack from source model files and optional MTP sidecar
[[nodiscard]] PackPlan plan_derived_pack(
    const ModelFiles& source,
    const ModelFiles* mtp_sidecar = nullptr,
    uint64_t byte_budget = max_pack_bytes
);

// Build derived pack with byte-exact inverse reconstruction proof across all tensors
[[nodiscard]] PackPlan build_derived_pack(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& output,
    uint64_t reserve_bytes = default_reserve_bytes,
    uint64_t chunk_bytes = default_chunk_bytes
);

// Build derived pack using an explicit directory free-space query callback
[[nodiscard]] PackPlan build_derived_pack_with_space_query(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& output,
    uint64_t reserve_bytes,
    uint64_t chunk_bytes,
    const std::function<uint64_t(int)>& query_available_bytes
);

// full: every start reads the whole pack, checks each tensor digest and
// compares the pack with the source files.
//
// local_seal: skips that work when the seal beside the pack (<stem>.seal)
// vouches for it. A seal vouches when it is in the current format, carries the
// pack's manifest hash, and records exactly the inode, size, modification
// time, change time and birth time the pack file has now. Such a seal is only
// written after a full verification of that file, so a match means: this file
// passed full verification and has not been written, replaced, renamed or had
// its metadata changed since. In that case no payload byte is read and the
// pack is not compared with the source files again.
//
// A seal also records which source files that verification compared the pack
// with: the target shards, the MTP sidecar, both or neither. It is accepted
// only by a start that has no source beyond those. A start that has one more,
// for example the MTP sidecar after a start without it, does the full
// verification and widens the seal.
//
// It is not a defence against a process running as the same user, which can
// rewrite both the pack and the seal. Only full mode checks the pack against
// the source files on every start.
//
// A missing seal, a seal in the earlier format, a malformed seal, a seal that
// is writable, multiply linked, owned by another user or not a regular file,
// and any differing field all mean the same thing: this start does the full
// verification. So any
// change to the pack file, including copying or restoring it, costs one full
// verification. When that verification passes a current seal is written
// (temporary file, then rename), so the following start is fast again. A seal
// name that is not a regular file, or is a file too large to be a seal, is
// left alone and is never used. A pack
// whose own name ends in .seal has no seal name: it is fully verified on every
// start and nothing is written beside it.
enum class TrustMode { full, local_seal };

// Adopt a derived pack under the given trust mode. AdoptedPack::local_seal
// reports what the seal was used for.
[[nodiscard]] AdoptedPack adopt_derived_pack(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& path,
    uint64_t chunk_bytes = default_chunk_bytes,
    uint32_t authentication_workers = 1,
    TrustMode trust_mode = TrustMode::full,
    bool require_full_census = true
);

// Check disk budget given available space, peak allocation, and required reserve
[[nodiscard]] bool check_disk_reserve(
    uint64_t available_bytes,
    uint64_t peak_additional_bytes,
    uint64_t reserve_bytes = default_reserve_bytes
);

} // namespace tokspur::derived_pack

namespace tokspur {
using derived_pack::is_cpu_pinned_embedding;
}

