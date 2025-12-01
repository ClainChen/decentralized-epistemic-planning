;Header and description

(define
    (domain grid)

    (:types
        agent survivor location
    )

    (:functions
        (agent_loc ?a - agent)
        (survivor_loc ?s - survivor)
        (movable ?a - agent)
        (sharable ?a - agent)
        (receivable ?a - agent)
        (loc_id ?l - location)
        (searched ?l - location)
        (shared ?s - survivor)
        (connected ?l1 ?l2 - location)
        (sharing ?a - agent)
        (share_lock)
    )

    (:action move
        :parameters (?self - agent ?l1 ?l2 - location)
        :precondition (
            (= (share_lock) 0)
            (= (movable ?self) 1)
            (= (agent_loc ?self) (loc_id ?l1))
            (= (connected ?l1 ?l2) 1)
        )
        :effect (
            (assign (agent_loc ?self) (loc_id ?l2))
            (assign (searched ?l2) 1)
        )
    )

    (:action share_survivor
        :parameters (?self - agent ?s - survivor ?l - location)
        :precondition (
            (= (share_lock) 0)
            (= (sharing ?self) 0)
            (= (sharable ?self) 1)
            (= (survivor_loc ?s) (loc_id ?l))
            (= (agent_loc ?self) (loc_id ?l))
        )
        :effect (
            (assign (shared ?s) 1)
            (assign (share_lock) 1)
            (assign (sharing ?self) 1)
        )
    )

    (:action share
        :parameters (?self - agent ?l - location)
        :precondition (
            (= (share_lock) 0)
            (= (sharing ?self) 0)
            (= (sharable ?self) 1)
            (= (agent_loc ?self) (loc_id ?l))
        )
        :effect (
            (assign (share_lock) 1)
            (assign (sharing ?self) 1)
        )
    )

    (:action quiet
        :parameters (?self - agent)
        :precondition (
            (= (share_lock) 1)
            (= (sharing ?self) 1)
        )
        :effect (
            (assign (share_lock) 0)
            (assign (sharing ?self) 0)
        )
    )

    (:action normal_stay
        :parameters (?self - agent)
        :precondition (
            (= (share_lock) 0)
        )
        :effect (

        )
    )

    (:action lock_stay
        :parameters (?self - agent)
        :precondition (
            (= (share_lock) 1)
            (= (sharing ?self) 0)
        )
        :effect (

        )
    )
    
    
    
    
)