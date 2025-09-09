(define 
    (domain grapevine)

    (:types
        secret agent
    )

    ;the sharing_lock will block all other update-actions until no secret is sharing
    (:functions
        (agent_loc ?a - agent)
        (secret_id ?s - secret)
        (shared_value ?s - secret)
        (shared_loc ?s - secret)
        (own ?a - agent)
        (sharing_lock)
        (agent_sharing ?a - agent)
    )

    ;define actions here
    (:action move_right
        :parameters (?self - agent)
        :precondition (
            (= (sharing_lock) 0)
            (= (agent_sharing ?self) none)
            (= (agent_loc ?self) 1)
        )
        :effect (
            (assign (agent_loc ?self) 2)
        )
    )
    
    (:action move_left
        :parameters (?self - agent)
        :precondition (
            (= (sharing_lock) 0)
            (= (agent_sharing ?self) none)
            (= (agent_loc ?self) 2)
        )
        :effect (
            (assign (agent_loc ?self) 1)
        )
    )

    (:action quiet
        :parameters (?self - agent ?s - secret)
        :precondition (
            (= (sharing_lock) 1)
            (= (agent_sharing ?self) (secret_id ?s))
            (!= (shared_loc ?s) 0)
        )
        :effect (
            (assign (sharing_lock) 0)
            (assign (agent_sharing ?self) none)
            (assign (shared_loc ?s) 0)
            (assign (shared_value ?s) t)
        )
    )
    

    (:action sharing_own_secret
        :parameters (?self - agent ?s - secret)
        :precondition (
            (= (sharing_lock) 0)
            (= (own ?self) (secret_id ?s))
            (= (agent_sharing ?self) none)
        )
        :effect (
            (assign (sharing_lock) 1)
            (assign (agent_sharing ?self) (secret_id ?s))
            (assign (shared_loc ?s) (agent_loc ?self))
            (assign (shared_value ?s) t)
        )
    )

    (:action lying_own_secret
        :parameters (?self - agent ?s - secret)
        :precondition (
            (= (sharing_lock) 0)
            (= (own ?self) (secret_id ?s))
            (= (agent_sharing ?self) none)
        )
        :effect (
            (assign (sharing_lock) 1)
            (assign (agent_sharing ?self) (secret_id ?s))
            (assign (shared_loc ?s) (agent_loc ?self))
            (assign (shared_value ?s) f)
        )
    )

    (:action sharing_others_secret
        :parameters (?self - agent ?s - secret)
        :precondition (
            (= (sharing_lock) 0)
            (= (own ?self) (secret_id ?s))
            (= (agent_sharing ?self) none)
            (!= (@ep ("b [?self]") (shared_value ?s)) ep.unknown)
        )
        :effect (
            (assign (sharing_lock) 1)
            (assign (agent_sharing ?self) (secret_id ?s))
            (assign (shared_loc ?s) (agent_loc ?self))
            (assign (shared_value ?s) (@ep ("b [?self]") (shared_value ?s)))
        )
    )
)