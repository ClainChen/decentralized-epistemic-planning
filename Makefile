.PHONY: help coin1 coin2 corridor2a1i_1 corridor2a2i_1 corridor2a2i_2 grapevine1 grapevine2 grapevine3 mapf1 mapf2 mapf3 matrix2a1i3r_1 matrix2a1i4r_1 clean

.DEFAULT_GOAL := help

args ?=

help:
	@echo Useful commands:
	@echo   make coin1         
	@echo   make coin2         
	@echo   make corridor2a1i_1
	@echo   make corridor2a2i_1
	@echo   make corridor2a2i_2
	@echo   make grapevine1    
	@echo   make grapevine2    
	@echo   make grapevine3    
	@echo   make mapf1         
	@echo   make mapf2         
	@echo   make matrix2a1i3r_1
	@echo   make matrix2a1i4r_1
	@echo   make clean         

coin1:
	python entrance.py \
		-d coin/domain.pddl \
		-p coin/problem1 \
		-ob coin.py \
		--strategy s-jbfs.py \
		--rules coin.py \
		$(args)

coin2:
	python entrance.py \
		-d coin/domain.pddl \
		-p coin/problem2 \
		-ob coin.py \
		--strategy s-jbfs.py \
		--rules coin.py \
		$(args)

corridor2a1i_1:
	python entrance.py \
		-d corridor/domain.pddl \
		-p corridor/2a1i_1 \
		-ob corridor.py \
		--strategy s-jbfs.py \
		--rules corridor.py \
		$(args)

corridor2a2i_1:
	python entrance.py \
		-d corridor/domain.pddl \
		-p corridor/2a2i_1 \
		-ob corridor.py \
		--strategy s-jbfs.py \
		--rules corridor.py \
		$(args)

corridor2a2i_2:
	python entrance.py \
		-d corridor/domain.pddl \
		-p corridor/2a2i_2 \
		-ob corridor.py \
		--strategy s-jbfs.py \
		--rules corridor.py \
		$(args)

grapevine1:
	python entrance.py \
		-d grapevine/domain.pddl \
		-p grapevine/problem1 \
		-ob grapevine.py \
		--strategy s-jbfs.py \
		--rules grapevine.py \
		$(args)

grapevine2:
	python entrance.py \
		-d grapevine/domain.pddl \
		-p grapevine/problem2 \
		-ob grapevine.py \
		--strategy s-jbfs.py \
		--rules grapevine.py \
		$(args)

grapevine3:
	python entrance.py \
		-d grapevine/domain.pddl \
		-p grapevine/problem3 \
		-ob grapevine.py \
		--strategy s-jbfs.py \
		--rules grapevine.py \
		$(args)

mapf1:
	python entrance.py \
		-d mapf/domain.pddl \
		-p mapf/problem1 \
		-ob mapf.py \
		--strategy s-jbfs.py \
		--rules mapf.py \
		$(args)

mapf2:
	python entrance.py \
		-d mapf/domain.pddl \
		-p mapf/problem2 \
		-ob mapf.py \
		--strategy s-jbfs.py \
		--rules mapf.py \
		$(args)

mapf3:
	python entrance.py \
		-d mapf/domain.pddl \
		-p mapf/problem3 \
		-ob mapf.py \
		--strategy s-jbfs.py \
		--rules mapf.py \
		$(args)

matrix2a1i3r_1:
	python entrance.py \
		-d matrix/domain.pddl \
		-p matrix/2a1i3r_1 \
		-ob matrix.py \
		--strategy s-jbfs.py \
		--rules matrix.py \
		$(args)

matrix2a1i4r_1:
	python entrance.py \
		-d matrix/domain.pddl \
		-p matrix/2a1i4r_1 \
		-ob matrix.py \
		--strategy s-jbfs.py \
		--rules matrix.py \
		$(args)

clean:
	rm -f *.pyc __pycache__/*.pyc