#include "llama.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/mman.h>
int main(int argc,char**argv){
  const char* model_path=argv[1]; int np=std::atoi(argv[2]);
  llama_backend_init();
  llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=0;
  llama_model* model=llama_model_load_from_file(model_path,mp);
  llama_context_params cp=llama_context_default_params(); cp.n_ctx=4096; cp.n_ubatch=512; cp.n_batch=2048; cp.n_threads=8;
  llama_context* ctx=llama_init_from_model(model,cp);
  llama_set_n_threads(ctx,8,8);
  std::vector<llama_token> pt(np); unsigned s=7;
  for(int i=0;i<np;i++){s=s*1103515245+12345;pt[i]=(llama_token)((s>>16)%2000+10);}
  for(int off=0;off<np;off+=512){int nch=(np-off)<512?(np-off):512; llama_decode(ctx,llama_batch_get_one(pt.data()+off,nch));}
  size_t sz=llama_state_get_size(ctx);
  auto t0=std::chrono::steady_clock::now();
  for(int r=0;r<3;r++){ llama_state_save_file(ctx,"/tmp/zct.bin",pt.data(),pt.size()); }
  auto t1=std::chrono::steady_clock::now();
  for(int r=0;r<3;r++){ std::vector<llama_token> toks(4096); size_t n=0; llama_state_load_file(ctx,"/tmp/zct.bin",toks.data(),toks.size(),&n); }
  auto t2=std::chrono::steady_clock::now();
  remove("/tmp/zct.bin");
  auto t3=std::chrono::steady_clock::now();
  int mfd=(int)syscall(319,"z",0); ftruncate(mfd,(off_t)sz);
  void* shm=mmap(nullptr,sz,PROT_READ|PROT_WRITE,MAP_SHARED,mfd,0);
  for(int r=0;r<3;r++){ llama_state_get_data(ctx,(uint8_t*)shm,sz); llama_state_set_data(ctx,(const uint8_t*)shm,sz); }
  auto t4=std::chrono::steady_clock::now();
  double fsave=std::chrono::duration<double,std::milli>(t1-t0).count()/3;
  double fload=std::chrono::duration<double,std::milli>(t2-t1).count()/3;
  double msave=std::chrono::duration<double,std::milli>(t4-t3).count()/3;
  fprintf(stderr,"state=%zu B | F file: save %.1fms + load %.1fms = %.1fms total | M memfd: get+set %.1fms total\n",sz,fsave,fload,fsave+fload,msave);
  auto d0=std::chrono::steady_clock::now();
  int tok=pt.back();
  for(int i=0;i<5;i++){ llama_decode(ctx,llama_batch_get_one(&tok,1)); }
  auto d1=std::chrono::steady_clock::now();
  double per=std::chrono::duration<double,std::milli>(d1-d0).count()/5;
  fprintf(stderr,"decode: %.1f ms/token (CPU) ; M handoff = %.2f ms = %.3fx one decode step\n",per,msave,msave/per);
  llama_free(ctx); llama_free_model(model); llama_backend_free();
  return 0;
}
