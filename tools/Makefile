# TODO: check pytorch version: 1.3.1, 1.4.0, 1.5.1, 1.6.0, 1.7.1, 1.8.1, 1.9.1, 1.10.0 and 1.10.1


CXX ?= g++

WGET ?= wget

# Note: OpenFst requires a relatively recent C++ comile with C++ 11 support,
# e.g. g++ >= 4.7, Apple clang >= 5.0 or LLVM clang >= 3.3.

OPENFST_VERSION ?= 1.6.7

# Default features configured for OpenFST; can be overwridden in the make command line.
OPENFST_CONFIGURE ?= --enable-static --enable-shared --enable-ngram-fsts

CPPFLAGS ?= -D_GLIBCXX_USE_CXX11_ABI=0
CXXFLAGS ?= -D_GLIBCXX_USE_CXX11_ABI=0

TH_VERSION := 1.10.1

# Set if binaies on CPU mode e.g. make CPU_ONLY=0
# If you don't have nvcc, this value will be set automatically
ifneq ($(shell which nvcc 2>/dev/null),)
CPU_ONLY :=
# Get CUDA version from nvcc
CUDA_VERSION := $(shell nvcc --version | grep "Cuda compilation tools" | cut -d" " -f5 | sed s/,//)
CUDA_VERSION_WITHOUT_DOT := $(strip $(subst .,,$(CUDA_VERSION)))
else
CPU_ONLY := 1
CUDA_VERSION :=
CUDA_VERSION_WITHOUT_DOT :=
endif

.PHONY: all clean

all: python sctk.done sph2pipe.done  ctc_crf

clean: openfst_cleaned ctc_crf_clean
	rm -rf *.done

python:  catex.done pytorch.done torch_optimizer.done matplotlib.done


############### Logging ###########################################
showenv:
ifeq ($(strip $(CPU_ONLY)),)
	@echo CUDA_VERSION=$(CUDA_VERSION)
else
	@echo Perfom on CPU mode: CPU_ONLY=$(CPU_ONLY)
endif
	@echo PYTHON=$(shell  command -v python3)
	@echo PYTHON_VERSION=$(shell  python3 --version)
	@echo TH_VERSION=$(TH_VERSION)
###################################################################
matplotlib.done:
	python3 -m pip install matplotlib
	touch matplotlib.done
sctk.done:
	./installers/install_sctk.sh
	touch sctk.done
sph2pipe.done:
	./installers/install_sph2pipe.sh
	touch sph2pipe.done

pytorch.done: 
	./installers/install_torch.sh "false" "${TH_VERSION}" "${CUDA_VERSION}"
	touch pytorch.done

catex.done: pytorch.done
	python3 -m pip install -e "..[train, test]" # Install with editable mode
	touch catex.done

torch_optimizer.done:
	./installers/install_torch_optimizer.sh
	touch torch_optimizer.done

############### openfst installation process ############################
openfst_cleaned:
	$(MAKE) -C openfst-$(OPENFST_VERSION) clean

.PHONY: openfst # so target will be made even though "exists".
openfst: openfst_compiled openfst-$(OPENFST_VERSION)/lib
	-rm -f openfst
	-ln -s openfst-$(OPENFST_VERSION) openfst

.PHONY: openfst_compiled
openfst_compiled: openfst-$(OPENFST_VERSION)/Makefile
	$(MAKE) -C openfst-$(OPENFST_VERSION) install MAKEOVERRIDES=

openfst-$(OPENFST_VERSION)/lib: | openfst-$(OPENFST_VERSION)/Makefile
	- cd openfst-$(OPENFST_VERSION) && \
	[ -d lib64 ] && [ ! -d lib ] && ln -s lib64 lib


# Add the -O flag to CXXFLAGS on cygwin as it can fix the compilation error
# "file too big."
ifeq ($(OSTYPE), cygwin)
  # Note: OSTYPE path is probably dead for latest cygwin64
  openfst_add_CXXFLAGS = -O -Wa,-mbig-obj
else ifeq ($(OS),Windows_NT)
  # This new OS path is confirmed working on Windows 10 / Cygwin64.
  openfst_add_CXXFLAGS = -O -Wa,-mbig-obj
else
  openfst_add_CXXFLAGS =
endif

openfst-$(OPENFST_VERSION)/Makefile: openfst-$(OPENFST_VERSION)
	cd openfst-$(OPENFST_VERSION)/ && \
	./configure --prefix=`pwd` $(OPENFST_CONFIGURE) CXX="$(CXX)" CPPFLAGS="$(CPPFLAGS)" CXXFLAGS="$(CXXFLAGS) $(openfst_add_CXXFLAGS)" LDFLAGS="$(LDFLAGS)" LIBS="-ldl"

openfst-$(OPENFST_VERSION): openfst-$(OPENFST_VERSION).tar.gz
	tar xozf openfst-$(OPENFST_VERSION).tar.gz

openfst-$(OPENFST_VERSION).tar.gz:
	$(WGET) -T 10 -t 3  --no-check-certificate https://www.openslr.org/resources/2/openfst-$(OPENFST_VERSION).tar.gz || \
	$(WGET) -T 10 -t 1 --no-check-certificate http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-$(OPENFST_VERSION).tar.gz;

######################################################################

############### ctc_crf installation process #########################

OPENFST_DIR=$(PWD)/openfst
.PHONY: ctc_crf gpu_den.done gpu_ctc.done path_weight.done
gpu_den.done: openfst
	cd ../catex/asr/crf/ctc_crf && \
	mkdir -p gpu_den/build && \
	cd gpu_den/build && \
	cmake -D openfst_dir:STRING=${OPENFST_DIR} .. && \
	make
	touch gpu_den.done

gpu_ctc.done:
	cd ../catex/asr/crf/ctc_crf && \
	mkdir -p gpu_ctc/build && \
	cd gpu_ctc/build && \
	cmake .. && \
	make
	touch gpu_ctc.done

path_weight.done: openfst
	cd ../catex/asr/crf/ctc_crf && \
	mkdir -p path_weight/build && \
	cd path_weight/build && \
	cmake -D openfst_dir:STRING=${OPENFST_DIR} .. && \
	make
	touch path_weight.done

ctc_crf: gpu_ctc.done gpu_den.done path_weight.done
	cd ../catex/asr/crf/ctc_crf && \
	python3 -m pip install -e .

ctc_crf_clean:
	rm gpu_den.done gpu_ctc.done path_weight.done
	cd ../catex/asr/crf/ctc_crf && \
	rm -rf gpu_{ctc,den}/build path_weight/build build/ \
		dist/ ctc_crf.egg-info/

######################################################################