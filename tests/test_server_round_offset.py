import unittest

from flwr.common import Code, GetPropertiesRes, Status

from fednnunet.server import MyStrategy


class FakeClient:
    def __init__(self, cid, logical_id):
        self.cid = cid
        self.logical_id = logical_id

    def get_properties(self, ins, timeout, group_id):
        return GetPropertiesRes(
            status=Status(code=Code(0), message=""),
            properties={"client_id": self.logical_id},
        )


class FakeClientManager:
    def __init__(self, clients):
        self.clients = {client.cid: client for client in clients}

    def all(self):
        return self.clients

    def num_available(self):
        return len(self.clients)


class ServerRoundOffsetTests(unittest.TestCase):
    def test_sampling_uses_stable_logical_client_ids(self):
        manager = FakeClientManager(
            [
                FakeClient("random-z", "302"),
                FakeClient("random-a", "303"),
                FakeClient("random-m", "301"),
            ]
        )
        strategy = MyStrategy("train", clients_per_round=1, total_rounds=120)
        selected = [
            strategy.sample_fit_clients(round_id, manager)[0].logical_id
            for round_id in (1, 2, 3, 4)
        ]
        self.assertEqual(selected, ["301", "302", "303", "301"])

    def test_resume_offset_continues_the_same_sampling_cycle(self):
        manager = FakeClientManager(
            [FakeClient("new-3", "303"), FakeClient("new-1", "301"), FakeClient("new-2", "302")]
        )
        strategy = MyStrategy(
            "train", clients_per_round=1, total_rounds=150, round_offset=120
        )
        logical_round = strategy.round_offset + 1
        selected = strategy.sample_fit_clients(logical_round, manager)
        self.assertEqual(selected[0].logical_id, "301")


if __name__ == "__main__":
    unittest.main()
