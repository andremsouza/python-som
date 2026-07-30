# Artifact safety

Loading a `.npz` written by `save_npz` cannot execute code. Loading a `.pkl` can. That is a
difference in kind rather than degree, and it is the reason this package ships an artifact format
at all: not to forbid `pickle`, but to make the safe path the obvious one so nobody has to reach
for the unsafe one.

What follows is the precise boundary, because "safer" without a boundary is not a claim anyone can
act on.

Safer than a pickle, and the difference is categorical rather than a matter of degree, but not
unconditionally safe, so here is exactly what is and is not guaranteed.

**Cannot happen:**

- **Code execution.** `allow_pickle=False` is passed explicitly, so a crafted object array is
  refused by NumPy, not unpickled. The test suite builds a payload that really would create a file,
  proves it executes when NumPy is *allowed* to unpickle it, then shows the loader refusing the same
  file with nothing created.
- **Loading an arbitrary function by name.** Names resolve only through the registries. Nothing from
  the file is imported, evaluated, or looked up in a module.
- **A partly-read file being treated as valid.** The member list, the metadata JSON, the format
  version, the weight dimensions and the weight shape against the saved config are all checked
  before a map is constructed. Every failure raises `ArtifactError`.

**Can still happen:**

- **Resource exhaustion.** An `.npz` is a zip, so a hostile file can be built to decompress to
  something enormous. Nothing here caps that.

So: treat an artifact from an untrusted source the way you would a JPEG from one. It cannot run
code, but it can still be malformed or oversized. Do not treat it as you would a signed archive.

If you need integrity as well, hash the file and record the digest alongside whatever references it;
that is outside what this format tries to do.

## Why the guarantee stops there

`allow_pickle=False` closes the code-execution path and nothing else. A zip archive can still be
crafted to decompress to something enormous, and nothing in this format caps that. Capping it
would mean inventing a size limit with no principled value, which is the kind of unsourced
constant this package removes when it finds one.

If you need integrity as well as safety, that is a different property and wants a different tool:
hash the file and record the digest wherever the artifact is referenced.

## How the claim is tested

The test suite does not assert that pickles are refused. It builds a payload with a `__reduce__`
that creates a file, proves the payload really executes when something unpickles it, and then
shows the loader refusing a file containing it with nothing created.

Proving liveness first is the part that matters. Without it the test would pass just as well
against an inert payload, and would be evidence of nothing.
