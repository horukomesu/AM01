#include "miniz.h"
#include "json.hpp"
#include "md5.h"
#include <pybind11/pybind11.h>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <cstdlib>
#include <cstring>
extern "C" {
struct Buffer { unsigned char* data; int size; };

static std::string md5_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if(!f) return {};
    md5_state st; md5_init(st);
    char buf[4096];
    while(f.read(buf,sizeof(buf)) || f.gcount()) {
        md5_process(st, buf, (uint32_t)f.gcount());
    }
    unsigned char out[16];
    md5_done(st,out);
    static const char* hex="0123456789abcdef";
    std::string s; s.reserve(32);
    for(int i=0;i<16;i++){s.push_back(hex[out[i]>>4]);s.push_back(hex[out[i]&15]);}
    return s;
}

static std::string md5_string(const std::string& s) {
    md5_state st; md5_init(st); md5_process(st, s.data(), (uint32_t)s.size()); unsigned char out[16]; md5_done(st,out); static const char* hex="0123456789abcdef"; std::string r; r.reserve(32); for(int i=0;i<16;i++){r.push_back(hex[out[i]>>4]); r.push_back(hex[out[i]&15]);} return r; }

bool save_ams(const char* filepath, const char* json_data, const char** image_paths, int image_count) {
    std::filesystem::path zip_path(filepath);
    std::unordered_map<std::string,std::string> old_hashes;
    if(std::filesystem::exists(zip_path)) {
        mz_zip_archive zip{}; memset(&zip,0,sizeof(zip));
        if(!mz_zip_reader_init_file(&zip, filepath, 0)) return false;
        int idx = mz_zip_reader_locate_file(&zip, "meta/hashmap.json", nullptr, 0);
        if(idx>=0){ size_t sz; char* buf=(char*)mz_zip_reader_extract_to_heap(&zip, idx,&sz,0); if(buf){ auto j=nlohmann::json::parse(std::string(buf,sz)); for(auto&[k,v]:j.items()) old_hashes[k]=v.get<std::string>(); mz_free(buf);} }
        mz_zip_reader_end(&zip);
    }
    std::unordered_map<std::string,std::string> new_hashes;
    new_hashes["scene.json"] = md5_string(json_data);
    for(int i=0;i<image_count;++i){new_hashes[std::string("images/")+std::filesystem::path(image_paths[i]).filename().string()] = md5_file(image_paths[i]);}

    mz_zip_archive zipw{}; memset(&zipw,0,sizeof(zipw));
    if(!mz_zip_writer_init_file(&zipw, filepath, 0)) return false;
    if(!mz_zip_writer_add_mem(&zipw, "scene.json", json_data, strlen(json_data), MZ_BEST_COMPRESSION)) return false;
    for(int i=0;i<image_count;++i){ const char* img=image_paths[i]; std::string name=std::string("images/")+std::filesystem::path(img).filename().string(); if(!mz_zip_writer_add_file(&zipw, name.c_str(), img, nullptr,0, MZ_BEST_COMPRESSION)) return false; }
    nlohmann::json j(old_hashes);
    for(auto&[k,v]:new_hashes) j[k]=v;
    std::string jstr=j.dump();
    mz_zip_writer_add_mem(&zipw, "meta/hashmap.json", jstr.data(), jstr.size(), MZ_BEST_COMPRESSION);
    mz_zip_writer_finalize_archive(&zipw); mz_zip_writer_end(&zipw); return true;
}

bool load_ams(const char* filepath, char** out_json_data, char*** out_image_names, unsigned char*** out_image_buffers, int* out_image_sizes, int* out_count) {
    mz_zip_archive zip; memset(&zip,0,sizeof(zip)); if(!mz_zip_reader_init_file(&zip, filepath, 0)) return false;
    int file_count=(int)mz_zip_reader_get_num_files(&zip); std::vector<std::string> names; std::vector<Buffer> buffers;
    char* json_data_ptr=nullptr;
    for(int i=0;i<file_count;i++){ mz_zip_archive_file_stat st; mz_zip_reader_file_stat(&zip,i,&st); std::string name=st.m_filename;
        if(name=="scene.json"){ size_t sz; json_data_ptr=(char*)mz_zip_reader_extract_to_heap(&zip,i,&sz,0); if(!json_data_ptr){ mz_zip_reader_end(&zip); return false; } }
        else if(name.rfind("images/",0)==0){ size_t sz; unsigned char* buf=(unsigned char*)mz_zip_reader_extract_to_heap(&zip,i,&sz,0); if(!buf){ mz_zip_reader_end(&zip); return false; } names.push_back(name.substr(7)); buffers.push_back({buf,(int)sz}); }
    }
    mz_zip_reader_end(&zip);
    *out_json_data=json_data_ptr; *out_count=(int)names.size();
    *out_image_names=(char**)malloc(names.size()*sizeof(char*));
    *out_image_buffers=(unsigned char**)malloc(names.size()*sizeof(unsigned char*));
    *out_image_sizes=(int*)malloc(names.size()*sizeof(int));
    for(size_t i=0;i<names.size();++i){ (*out_image_names)[i]=strdup(names[i].c_str()); (*out_image_buffers)[i]=buffers[i].data; (*out_image_sizes)[i]=buffers[i].size; }
    return true;
}

void free_loaded_ams(char* json_data, char** image_names, unsigned char** image_buffers, int image_count){
    if(json_data) mz_free(json_data);
    for(int i=0;i<image_count;++i){ if(image_names) free(image_names[i]); if(image_buffers) mz_free(image_buffers[i]); }
    free(image_names); free(image_buffers);
}
}

static pybind11::object load_ams_wrapper(const char* filepath){ char* json_data; char** names; unsigned char** bufs; int* sizes; int count; if(!load_ams(filepath,&json_data,&names,&bufs,&sizes,&count)) throw std::runtime_error("load failed"); pybind11::dict images; for(int i=0;i<count;++i){ pybind11::bytes data((const char*)bufs[i],sizes[i]); images[names[i]]=data; } pybind11::dict result; result["scene"]=pybind11::str(json_data); result["images"]=images; free_loaded_ams(json_data,names,bufs,count); return result; }

PYBIND11_MODULE(Filesystem, m){ m.def("save_ams", &save_ams, "Save AMS file"); m.def("load_ams", &load_ams_wrapper, "Load AMS file"); }

