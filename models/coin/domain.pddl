;Header and description

(define
    (domain coin)

    (:types
        coin agent
    )

    (:functions
        (peeking ?a - agent)
        (coin ?c - coin)
    )

    (:action peek
        :parameters (?self - agent)
        :precondition (
            (= (peeking ?self) 0)
        )
        :effect (
            (assign (peeking ?self) 1)
        )
    )

    (:action return
        :parameters (?self - agent)
        :precondition (
            (= (peeking ?self) 1)
        )
        :effect (
            (assign (peeking ?self) 0)
        )
    )

    ; Here, the problem change to only the 
    (:action flip_up
        :parameters (?self - agent ?c - coin)
        :precondition (
            (= (peeking ?self) 1)
            (= (coin ?c) 0)
        )
        :effect (
            (assign (coin ?c) 1)
        )
    )

    (:action flip_down
        :parameters (?self - agent ?c - coin)
        :precondition (
            (= (peeking ?self) 1)
            (= (coin ?c) 1)
        )
        :effect (
            (assign (coin ?c) 0)
        )
    )
)