def _new_local_repository_env_impl(repository_ctx):
  attr = repository_ctx.attr
  repo_path = repository_ctx.path(repository_ctx.os.environ[attr.path_var_name])
  build_file_label = attr.build_file # Label(attr.build_file)
  files = repo_path.readdir()
  for f in files:
    repository_ctx.symlink(f, f.basename)
  repository_ctx.symlink(build_file_label, "BUILD.bazel")

new_local_repository_env = repository_rule(
    implementation = _new_local_repository_env_impl,
    local = True,
    attrs = {
      "path_var_name": attr.string(),
      "build_file": attr.label(),
    }
)