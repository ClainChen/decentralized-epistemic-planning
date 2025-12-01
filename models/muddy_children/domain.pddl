;Header and description

(define
    (domain muddy_children)

    (:types
        agent
    )

    (:functions
        (number_of_questions)
        (muddy ?a - agent)
        (is_type ?a - agent)
        (shouted)

    )

    (:action say_yes
        :parameters (?self - agent)
        :precondition (
            (= (is_type ?self) child)
            (= (shouted) 0)
            (= (@ep ("b [?self]") (= (muddy ?self) 1)) ep.true)
        )
        :effect (
            (assign (shouted) 1)
        )
    )

    (:action ask
        :parameters (?self - agent)
        :precondition (
            (= (is_type ?self) teacher)
        )
        :effect (
            (increase (number_of_questions) 1)
        )
    )
    
)