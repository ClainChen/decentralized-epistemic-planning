;Header and description

(define
    (domain consecutive_number)

    (:types
        agent
    )

    (:functions
        (number_range_min)
        (number_range_max)
        (agent_id ?a - agent)
        (agent_number ?a - agent)
        (agent_know ?a - agent)
    )

    (:action say_unknown
        :parameters (?self - agent)
        :precondition (

        )
        :effect (
            (increase (number_range_min) 1)
            (decrease (number_range_max) 1)
        )
    )

    (:action say_known
        :parameters (?self ?b - agent)
        :precondition (
            (!= (agent_id ?self) (agent_id ?b))
            (!= (@ep ("b [?self]") (agent_number ?b)) ep.unknown)
        )
        :effect (
            (assign (agent_know ?self) 1)
        )
    )
    
)