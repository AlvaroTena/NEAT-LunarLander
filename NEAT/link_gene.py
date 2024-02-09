class LinkId:
    def __init__(self, input_id: int, output_id: int):
        self.input_id = input_id
        self.output_id = output_id

    def __eq__(self, other):
        if isinstance(other, LinkId):
            return self.input_id == other.input_id and self.output_id == other.output_id
        else:
            return False

    def __hash__(self):
        return hash(self.input_id) ^ (hash(self.output_id) << 1)


class LinkGene:

    def __init__(self, link_id: LinkId, weight: float, is_enabled: bool):
        self.link_id = link_id
        self.weight = weight
        self.is_enabled = is_enabled
