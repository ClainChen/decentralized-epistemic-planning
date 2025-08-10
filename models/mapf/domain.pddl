;Header and description

(define
    (domain mapf)

    (:types
        agent room
    )

    (:functions
        (agent_at ?a - agent ?r - room)
        (connected ?r1 ?r2 - room)
        (room_has_agent ?r - room)
    )

    (:action move
        :parameters (?self - agent ?from ?to - room)
        :precondition (
            (= (connected ?from ?to) 1)
            (= (agent_at ?self ?from) 1)
            (= (room_has_agent ?to) 0)
            (= (room_has_agent ?from) 1)
        )
        :effect (
            (assign (agent_at ?self ?from) 0)
            (assign (agent_at ?self ?to) 1)
            (assign (room_has_agent ?from) 0)
            (assign (room_has_agent ?to) 1)
        )
    )
)