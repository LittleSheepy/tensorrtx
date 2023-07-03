#pragma once
#include <dirent.h>
#include <fstream>
#include <vector>
#include "types.h"

DLL_EXPORT int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names);